/*
 * Levy Console page.
 *
 * Polls GET /console/traces?after=<version> and draws what each console request
 * did: the path it took through the real topology, where each step ran, how
 * long each step cost, why the cache decided what it decided, and what a hit
 * saved. It computes nothing about the cache itself: similarity, cosine, cost,
 * location and attribution all come from the server's trace.
 */
(function () {
  'use strict';

  var POLL_MS = 250;
  var HOP_MS = 260;
  var MAX_ROWS = 500;

  var OUT = {
    exact_hit: { marker: 'exact', color: '#6EE7B7', tint: 'rgba(110,231,183,0.12)', chip: 'EXACT', badge: 'Exact hit' },
    semantic_hit: { marker: 'semantic', color: '#7CB7FF', tint: 'rgba(124,183,255,0.12)', chip: 'SEM', badge: 'Semantic hit' },
    miss: { marker: 'miss', color: '#F5A524', tint: 'rgba(245,165,36,0.12)', chip: 'MISS', badge: 'Miss' },
    error: { marker: 'error', color: '#FF7A7A', tint: 'rgba(255,122,122,0.12)', chip: 'ERR', badge: 'Error' },
    in_flight: { marker: 'miss', color: '#F5A524', tint: 'rgba(245,165,36,0.08)', chip: 'LIVE', badge: 'In flight' }
  };
  var NODE_IDS = ['n-client', 'n-router', 'n-exact', 'n-semantic', 'n-llm'];
  var NODE_X = [52, 170, 292, 418, 576];
  var STAGE_NODE = { router: 1, exact_lookup: 2, embed: 3, index_search: 3, llm_call: 4, store: 4 };
  var REACH = { exact_hit: 3, semantic_hit: 4, miss: 5 };
  var SPAN = {
    exact_lookup: ['Exact lookup', '#6EE7B7'],
    embed: ['Embed', '#7CB7FF'],
    index_search: ['Index search', '#A5B4FC'],
    llm_call: ['Provider call', '#F5A524'],
    store: ['Store', '#98A2B0']
  };
  var SETTINGS = ['threshold', 'embedding_provider', 'embedding_model', 'llm_provider', 'llm_model',
    'exact_cache', 'semantic_cache', 'vector_index_backend', 'cache_store'];
  var PROVIDER_NAMES = { anthropic: 'Anthropic', ollama: 'Ollama' };

  var state = {
    epoch: null, version: 0, traces: new Map(), order: [],
    stats: null, spent: 0,
    selected: null, follow: true, anim: null, animFor: null
  };

  // -- helpers ---------------------------------------------------------------

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function fmtMs(x) {
    if (x === null || x === undefined) return '—';
    if (x < 0.1) return x.toFixed(3) + ' ms';
    if (x < 10) return x.toFixed(2) + ' ms';
    if (x < 1000) return x.toFixed(1) + ' ms';
    return Math.round(x).toLocaleString('en-US') + ' ms';
  }

  function fmtUsd(x) {
    if (x === null || x === undefined) return 'no price configured';
    if (x === 0) return '$0';
    return '$' + x.toFixed(x < 0.01 ? 5 : 2);
  }

  function fmtTime(epochSeconds) {
    var d = new Date(epochSeconds * 1000);
    var pad = function (n, w) { return String(n).padStart(w || 2, '0'); };
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()) + '.' + pad(d.getMilliseconds(), 3);
  }

  function hostOf(url) {
    try { return new URL(url).host; } catch (e) { return url || ''; }
  }

  function kind(t) { return t.status === 'in_flight' ? 'in_flight' : t.outcome; }

  function elapsedMs(t) {
    if (t.status !== 'in_flight') return t.total_ms;
    return Math.max(0, Date.now() - t.arrived_at * 1000);
  }

  function spanTotal(t, name) {
    return (t.spans || []).filter(function (s) { return s.name === name; })
      .reduce(function (a, s) { return a + s.duration_ms; }, 0);
  }

  function lookupMs(t) {
    if (typeof t.lookup_ms === 'number' && t.lookup_ms > 0) return t.lookup_ms;
    return spanTotal(t, 'exact_lookup') + spanTotal(t, 'embed') + spanTotal(t, 'index_search');
  }

  function llmLocation(t) {
    if (t.llm && t.llm.location) return t.llm.location;
    return t.config.llm_provider === 'ollama' ? 'local' : 'remote';
  }

  function reachOf(t) {
    var k = kind(t);
    if (REACH[k]) return REACH[k];
    var stage = k === 'error' ? (t.error && t.error.stage) : t.current_stage;
    return (STAGE_NODE[stage] || 1) + 1;
  }

  // -- polling -----------------------------------------------------------------

  function setConnected(ok) {
    var live = $('live');
    live.classList.toggle('ok', ok);
    live.classList.toggle('down', !ok);
    $('live-text').textContent = ok ? 'live · ' + location.host : 'disconnected · retrying';
  }

  function resetHistory() {
    state.traces = new Map();
    state.order = [];
    state.selected = null;
    stopAnimation();
  }

  function poll() {
    fetch('/console/traces?after=' + state.version, { cache: 'no-store' })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        if (state.epoch !== null && data.epoch !== state.epoch) {
          resetHistory();
          state.epoch = data.epoch;
          state.version = 0;
          setTimeout(poll, 0);
          return;
        }
        state.epoch = data.epoch;
        state.stats = data.stats;
        state.spent = data.spent_usd;
        data.traces.forEach(function (t) {
          var prev = state.traces.get(t.id);
          state.traces.set(t.id, t);
          if (!prev) {
            state.order.push(t.id);
            while (state.order.length > MAX_ROWS) state.traces.delete(state.order.shift());
          }
          onTraceUpdate(t, prev);
        });
        state.version = Math.max(state.version, data.version);
        setConnected(true);
        render();
        setTimeout(poll, POLL_MS);
      })
      .catch(function () {
        setConnected(false);
        setTimeout(poll, 1000);
      });
  }

  function onTraceUpdate(t, prev) {
    if (!prev && state.follow) {
      state.selected = t.id;
      play(t, true);
      return;
    }
    if (t.id !== state.selected) return;
    var stageChanged = prev && prev.current_stage !== t.current_stage;
    var finished = prev && prev.status === 'in_flight' && t.status !== 'in_flight';
    if (finished) play(t, false, prev);
    else if (stageChanged && t.status === 'in_flight') play(t, false, prev);
  }

  // -- packet animation --------------------------------------------------------

  function stopAnimation() {
    if (state.anim) cancelAnimationFrame(state.anim);
    state.anim = null;
    var packet = $('packet');
    if (packet) packet.classList.remove('on');
  }

  function pathPoints(t, fromNode) {
    var reach = reachOf(t);
    var pts = [];
    var i;
    for (i = fromNode; i < reach; i++) pts.push({ x: NODE_X[i], y: 112 });
    if (t.status === 'in_flight') return pts;
    var last = reach - 1;
    pts.push({ x: NODE_X[last], y: 136 });
    for (i = last - 1; i >= 0; i--) pts.push({ x: NODE_X[i], y: 136 });
    return pts;
  }

  function play(t, fromStart, prev) {
    stopAnimation();
    var startNode = 0;
    if (!fromStart && prev && prev.status === 'in_flight') startNode = Math.max(0, reachOf(prev) - 1);
    var pts = pathPoints(t, startNode);
    if (pts.length === 0) return;
    var packet = $('packet');
    packet.style.fill = OUT[kind(t)].color;
    packet.setAttribute('cx', pts[0].x);
    packet.setAttribute('cy', pts[0].y);
    packet.classList.add('on');
    var t0 = null;
    var parked = t.status === 'in_flight';
    function step(ts) {
      if (t0 === null) t0 = ts;
      var seg = Math.floor((ts - t0) / HOP_MS);
      if (seg >= pts.length - 1) {
        var end = pts[pts.length - 1];
        packet.setAttribute('cx', end.x - (parked ? 72 : 0));
        packet.setAttribute('cy', end.y);
        state.anim = null;
        if (!parked) setTimeout(function () { if (!state.anim) packet.classList.remove('on'); }, 400);
        return;
      }
      var f = ((ts - t0) % HOP_MS) / HOP_MS;
      var a = pts[seg], b = pts[seg + 1];
      packet.setAttribute('cx', a.x + (b.x - a.x) * f);
      packet.setAttribute('cy', a.y + (b.y - a.y) * f);
      state.anim = requestAnimationFrame(step);
    }
    state.anim = requestAnimationFrame(step);
  }

  // -- rendering ---------------------------------------------------------------

  function render() {
    renderSummary();
    renderHistory();
    var t = state.selected ? state.traces.get(state.selected) : null;
    renderMap(t);
    renderDetail(t);
  }

  function renderSummary() {
    var s = state.stats || { served: 0, exact_hits: 0, semantic_hits: 0, misses: 0, hit_rate: 0 };
    var all = Array.from(state.traces.values());
    var flying = all.filter(function (t) { return t.status === 'in_flight'; }).length;
    var errors = all.filter(function (t) { return t.outcome === 'error'; }).length;
    var notes = [];
    if (flying) notes.push(flying + ' in flight');
    if (errors) notes.push(errors + (errors === 1 ? ' error' : ' errors'));
    $('s-served').textContent = String(s.served);
    $('s-served-note').textContent = notes.length ? notes.join(' · ') : (s.served ? 'no errors' : 'no requests yet');
    $('s-hit').textContent = Math.round(s.hit_rate * 100) + ' %';
    $('s-hit-bar').style.width = (s.hit_rate * 100).toFixed(1) + '%';
    $('s-mix').textContent = s.exact_hits + ' exact · ' + s.semantic_hits + ' semantic · ' + s.misses + ' miss';
    $('s-spent').textContent = fmtUsd(state.spent || 0);
    $('s-spent-note').textContent = 'USD, at the prices in .env';
    var ms = 0, tokens = 0, usd = 0;
    all.forEach(function (t) {
      if (!t.savings) return;
      ms += t.savings.llm_ms || 0;
      tokens += t.savings.total_tokens || 0;
      usd += t.savings.cost_usd || 0;
    });
    $('s-avoided').textContent = (ms / 1000).toFixed(1) + ' s';
    $('s-avoided-note').textContent = tokens.toLocaleString('en-US') + ' tokens · ' + fmtUsd(usd);
  }

  function renderHistory() {
    var list = $('rows');
    list.textContent = '';
    $('rows-empty').hidden = state.order.length > 0;
    for (var i = state.order.length - 1; i >= 0; i--) {
      var t = state.traces.get(state.order[i]);
      if (!t) continue;
      var o = OUT[kind(t)];
      var li = el('li');
      var btn = el('button', 'row');
      btn.type = 'button';
      btn.setAttribute('aria-current', t.id === state.selected ? 'true' : 'false');
      btn.style.borderColor = t.id === state.selected ? o.color : 'transparent';
      btn.appendChild(el('span', 'row-num', '#' + t.seq));
      var top = el('span', 'row-top');
      var chip = el('span', 'chip', o.chip);
      chip.style.color = o.color;
      chip.style.background = o.tint;
      top.appendChild(chip);
      top.appendChild(el('span', 'prov', t.config.llm_provider));
      top.appendChild(el('span', 'lat', fmtMs(elapsedMs(t)) + (t.status === 'in_flight' ? '…' : '')));
      btn.appendChild(top);
      btn.appendChild(el('span', 'row-prompt', t.prompt));
      btn.addEventListener('click', select.bind(null, t.id));
      li.appendChild(btn);
      list.appendChild(li);
    }
  }

  function select(id) {
    state.selected = id;
    setFollow(false);
    var t = state.traces.get(id);
    if (t) play(t, true);
    render();
  }

  function setFollow(on) {
    state.follow = on;
    $('follow').setAttribute('aria-pressed', on ? 'true' : 'false');
    $('follow').textContent = on ? 'Following live' : 'Follow live';
  }

  function setEdge(id, color, marker, dashed) {
    var e = $(id);
    e.style.stroke = color;
    e.setAttribute('marker-end', 'url(#head-' + marker + ')');
    e.style.strokeDasharray = dashed ? '4 4' : 'none';
  }

  function renderMap(t) {
    $('host').textContent = location.host;
    if (!t) {
      ['f1', 'f2', 'f3', 'f4', 'r1', 'r2', 'r3', 'r4'].forEach(function (id) { setEdge(id, '#2A313C', 'idle', false); });
      return;
    }
    var k = kind(t);
    var o = OUT[k];
    var reach = reachOf(t);
    var cfg = t.config;
    var flying = t.status === 'in_flight';
    for (var i = 1; i <= 4; i++) {
      var fwd = i < reach;
      setEdge('f' + i, fwd ? o.color : '#2A313C', fwd ? o.marker : 'idle', false);
      var back = !flying && i < reach;
      setEdge('r' + i, back ? o.color : '#2A313C', back ? o.marker : 'idle', k === 'error' && back);
    }
    NODE_IDS.forEach(function (id, idx) {
      var rect = $(id).querySelector('rect');
      var active = idx < reach;
      rect.style.stroke = active ? o.color : '#2A313C';
      rect.style.fill = idx === reach - 1 ? o.tint : '';
    });
    $('n-exact').classList.toggle('off', !cfg.exact_cache);
    $('n-semantic').classList.toggle('off', !cfg.semantic_cache);
    var after = t.cache_after;
    $('exact-sub').textContent = cfg.exact_cache
      ? cfg.cache_store + (after ? ' · ' + after.exact_entries : '')
      : 'off · skipped';
    $('sem-sub').textContent = cfg.semantic_cache
      ? (after ? after.index_size + (after.index_size === 1 ? ' vector' : ' vectors') : (t.semantic ? t.semantic.index_size + ' vectors' : '…'))
      : 'off · skipped';
    $('llm-name').textContent = PROVIDER_NAMES[cfg.llm_provider] || cfg.llm_provider;
    $('llm-sub').textContent = cfg.llm_model.length > 18 ? cfg.llm_model.slice(0, 17) + '…' : cfg.llm_model;

    var remote = llmLocation(t) === 'remote';
    $('zone-local').setAttribute('width', remote ? '475' : '655');
    $('zone-net').style.display = remote ? '' : 'none';
    $('stored-exact').classList.toggle('on', k === 'miss' && cfg.exact_cache);
    $('stored-sem').classList.toggle('on', k === 'miss' && cfg.semantic_cache);
    $('wait-rings').classList.toggle('on', flying && t.current_stage === 'llm_call');

    $('foot-left').textContent = cfg.semantic_cache
      ? cfg.embedding_model + ' · ' + cfg.vector_index_backend + ' · τ ' + cfg.threshold
      : 'semantic cache off';
    $('foot-right').textContent = (t.llm ? hostOf(t.llm.endpoint) + ' · ' : '') + cfg.llm_model;
    $('map-title').textContent = 'Where request #' + t.seq + ' went';
    $('map-config').textContent = cfg.llm_provider + ' · τ ' + cfg.threshold + ' · ' + cfg.cache_store;

    var story = $('story');
    story.textContent = '';
    story.style.background = o.tint;
    var b = el('b', null, o.badge + '. ');
    b.style.color = o.color;
    story.appendChild(b);
    story.appendChild(document.createTextNode(storyText(t)));
  }

  function storyText(t) {
    var k = kind(t);
    var where = t.llm ? hostOf(t.llm.endpoint) : (PROVIDER_NAMES[t.config.llm_provider] || t.config.llm_provider);
    var local = llmLocation(t) === 'local';
    if (k === 'exact_hit') return 'Answered from the exact cache in ' + fmtMs(t.total_ms) + '. Same prompt, byte for byte; the request never left this machine.';
    if (k === 'semantic_hit') return 'Answered from the semantic cache in ' + fmtMs(t.total_ms) + '. It matched a stored prompt at similarity ' + t.semantic.similarity.toFixed(4) + ', above τ ' + t.config.threshold + ', and never left this machine.';
    if (k === 'miss') return (t.config.semantic_cache ? 'Neither cache could answer.' : 'The exact cache could not answer and the semantic cache was off for this request.') + ' Sent to ' + where + (local ? ' on this machine' : '') + ', back in ' + fmtMs(t.total_ms) + ', then stored.';
    if (k === 'error') return 'Failed at ' + (t.error.stage || 'the router') + ' after ' + fmtMs(t.total_ms) + '. The client got a structured ' + t.error.kind + ' error and nothing was cached.';
    if (t.current_stage === 'llm_call') return 'Neither cache could answer. Waiting on ' + where + ': ' + fmtMs(elapsedMs(t)) + ' so far.';
    return 'Running: ' + (SPAN[t.current_stage] ? SPAN[t.current_stage][0].toLowerCase() : t.current_stage) + '…';
  }

  function renderDetail(t) {
    var detail = $('detail');
    detail.hidden = !t;
    if (!t) return;
    var k = kind(t);
    var o = OUT[k];

    var badge = $('d-badge');
    badge.textContent = o.badge.toUpperCase();
    badge.style.color = o.color;
    badge.style.background = o.tint;
    $('d-total').textContent = t.status === 'in_flight' ? fmtMs(elapsedMs(t)) + ' and counting' : fmtMs(t.total_ms);
    $('d-meta').textContent = '#' + t.seq + ' · ' + fmtTime(t.arrived_at) + ' · ' + t.id.slice(0, 8);
    $('d-prompt').textContent = '“' + t.prompt + '”';

    renderConfig(t);
    renderSpans(t);
    renderSemantic(t);
    renderCost(t);
    $('d-answer').textContent = t.answer || (t.status === 'in_flight' ? 'Waiting for the provider. The answer appears here when the call returns.' : 'No answer: nothing was cached.');
    renderIdentity(t);
  }

  function renderConfig(t) {
    var dl = $('d-config');
    dl.textContent = '';
    var fromRequest = t.config_from_request || [];
    SETTINGS.forEach(function (name) {
      var req = fromRequest.indexOf(name) !== -1;
      dl.appendChild(el('dt', null, name));
      dl.appendChild(el('dd', req ? 'req' : 'def', String(t.config[name])));
      dl.appendChild(el('dd', 'src ' + (req ? 'req' : 'def'), req ? 'request' : 'default'));
    });
  }

  function renderSpans(t) {
    var box = $('d-spans');
    box.textContent = '';
    var total = elapsedMs(t) || 0.001;
    var spans = (t.spans || []).slice();
    if (t.status === 'in_flight' && t.current_stage && SPAN[t.current_stage]) {
      var last = spans.reduce(function (a, s) { return Math.max(a, s.start_ms + s.duration_ms); }, 0);
      spans.push({ name: t.current_stage, start_ms: last, duration_ms: total - last, running: true });
    }
    if (k(t) === 'error' && t.error && SPAN[t.error.stage]) {
      var done = spans.reduce(function (a, s) { return Math.max(a, s.start_ms + s.duration_ms); }, 0);
      if (!spans.some(function (s) { return s.name === t.error.stage; })) {
        spans.push({ name: t.error.stage, start_ms: done, duration_ms: Math.max(0, t.total_ms - done), failed: true });
      }
    }
    spans.forEach(function (s) {
      var label = SPAN[s.name] || [s.name, '#98A2B0'];
      var row = el('div', 'span-row');
      row.appendChild(el('span', 'span-name', label[0]));
      var track = el('div', 'span-track');
      var bar = el('div', 'span-bar' + (s.running ? ' running' : ''));
      bar.style.left = (s.start_ms / total * 100).toFixed(3) + '%';
      bar.style.width = (s.duration_ms / total * 100).toFixed(3) + '%';
      bar.style.background = s.failed || (s.name === 'llm_call' && k(t) === 'error') ? '#FF7A7A' : label[1];
      track.appendChild(bar);
      row.appendChild(track);
      var value = s.running ? 'running…' : (s.failed ? 'failed · ' : '') + fmtMs(s.duration_ms);
      row.appendChild(el('span', 'span-val', value));
      box.appendChild(row);
    });
    $('d-ratio').textContent = ratioText(t);
  }

  function k(t) { return kind(t); }

  function ratioText(t) {
    var kk = kind(t);
    var lk = lookupMs(t);
    if ((kk === 'exact_hit' || kk === 'semantic_hit') && t.savings && t.savings.llm_ms) {
      return 'Served in ' + fmtMs(t.total_ms) + ' instead of a ' + fmtMs(t.savings.llm_ms) + ' provider call: ' +
        Math.round(t.savings.llm_ms / t.total_ms).toLocaleString('en-US') + '× faster.';
    }
    if (kk === 'miss' && t.llm && t.llm.duration_ms) {
      return 'Lookup overhead ' + fmtMs(lk) + ', ' + (lk / t.llm.duration_ms * 100).toFixed(2) + ' % of the ' + fmtMs(t.llm.duration_ms) + ' provider call it could not avoid.';
    }
    if (kk === 'error') return 'Failed after ' + fmtMs(t.total_ms) + ' at ' + (t.error.stage || 'the router') + '.';
    if (kk === 'in_flight') return 'Lookup finished in ' + fmtMs(lk) + '. The provider call is still open.';
    return '';
  }

  function cell(key, value, color) {
    var c = el('div', 'cell');
    c.appendChild(el('span', 'cell-k', key));
    var v = el('span', 'cell-v', value);
    if (color) v.style.color = color;
    c.appendChild(v);
    return c;
  }

  function renderSemantic(t) {
    var box = $('d-sem');
    box.textContent = '';
    var skipped = (t.skipped || []).indexOf('semantic_cache') !== -1;
    var sem = t.semantic;
    if (skipped) {
      box.appendChild(el('p', 'well', 'The semantic cache was switched off for this request: no embedding, no index search.'));
      return;
    }
    if (kind(t) === 'exact_hit') {
      box.appendChild(el('p', 'well', 'Exact match on the SHA-256 of the prompt. The semantic cache was not consulted.'));
      return;
    }
    if (!sem) {
      box.appendChild(el('p', 'well', t.status === 'in_flight' ? 'Looking up…' : 'The request did not reach the semantic cache.'));
      return;
    }
    if (sem.index_size === 0 || sem.similarity === null) {
      box.appendChild(el('p', 'well', 'The semantic index was empty: there was nothing to compare against, so no embedding was computed for the lookup.'));
      return;
    }
    var color = sem.hit ? '#7CB7FF' : '#F5A524';
    var near = el('div', 'sem-near');
    near.appendChild(el('span', 'cell-k', 'Nearest cached prompt'));
    near.appendChild(el('span', 'well', '“' + sem.nearest_prompt + '”'));
    box.appendChild(near);

    var gauge = el('div', 'gauge');
    gauge.setAttribute('role', 'img');
    gauge.setAttribute('aria-label', 'Similarity ' + sem.similarity.toFixed(4) + ' against threshold ' + sem.threshold.toFixed(2));
    gauge.appendChild(el('div', 'g-track'));
    var pass = el('div', 'g-pass');
    pass.style.left = (sem.threshold * 100) + '%';
    gauge.appendChild(pass);
    var tau = el('div', 'g-tau');
    tau.style.left = (sem.threshold * 100) + '%';
    gauge.appendChild(tau);
    var dot = el('div', 'g-sim');
    dot.style.left = (sem.similarity * 100).toFixed(2) + '%';
    dot.style.background = color;
    gauge.appendChild(dot);
    var zero = el('span', 'g-lab', '0');
    zero.style.left = '0';
    gauge.appendChild(zero);
    var one = el('span', 'g-lab', '1');
    one.style.right = '0';
    gauge.appendChild(one);
    var tl = el('span', 'g-lab tau', 'τ ' + sem.threshold.toFixed(2));
    tl.style.left = (sem.threshold * 100) + '%';
    gauge.appendChild(tl);
    box.appendChild(gauge);

    var cells = el('div', 'cells');
    cells.appendChild(cell('L2 distance', sem.distance.toFixed(3)));
    cells.appendChild(cell('Similarity', sem.similarity.toFixed(4), color));
    cells.appendChild(cell('Cosine', sem.cosine.toFixed(4)));
    box.appendChild(cells);

    var f = el('pre', 'well formula',
      'similarity = 1 / (1 + ' + sem.distance.toFixed(3) + ') = ' + sem.similarity.toFixed(4) + '\n' +
      sem.similarity.toFixed(4) + (sem.hit ? ' ≥ ' : ' < ') + 'τ ' + sem.threshold.toFixed(2) + '  →  ' + (sem.hit ? 'HIT' : 'MISS') + '\n' +
      'cosine = 1 − d² / 2 = ' + sem.cosine.toFixed(4));
    box.appendChild(f);
  }

  function renderCost(t) {
    var kk = kind(t);
    var box = $('d-cost');
    box.textContent = '';
    var origin = $('d-origin');
    origin.hidden = true;
    var err = $('d-error');
    err.hidden = true;
    if (kk === 'exact_hit' || kk === 'semantic_hit') {
      $('d-cost-title').textContent = 'WHAT THIS HIT AVOIDED';
      var s = t.savings;
      if (!s) {
        box.appendChild(cell('Origin', 'stored before this history'));
        return;
      }
      box.appendChild(cell('Provider time', fmtMs(s.llm_ms)));
      box.appendChild(cell('Tokens', s.total_tokens !== null && s.total_tokens !== undefined ? s.total_tokens.toLocaleString('en-US') : '—'));
      box.appendChild(cell('Cost', fmtUsd(s.cost_usd)));
      var src = state.traces.get(s.origin_id);
      if (src) {
        origin.hidden = false;
        origin.textContent = 'Open #' + src.seq + ', the miss that stored this answer';
        origin.onclick = select.bind(null, src.id);
      }
      return;
    }
    if (kk === 'miss') {
      $('d-cost-title').textContent = 'WHAT THIS CALL COST';
      var l = t.llm;
      box.appendChild(cell('Input tokens', l.input_tokens !== null ? String(l.input_tokens) : '—'));
      box.appendChild(cell('Output tokens', l.output_tokens !== null ? String(l.output_tokens) : '—'));
      box.appendChild(cell('Cost', fmtUsd(l.cost_usd)));
      box.appendChild(cell('Stop reason', l.stop_reason || '—'));
      return;
    }
    if (kk === 'error') {
      $('d-cost-title').textContent = 'WHAT FAILED';
      box.appendChild(cell('Kind', t.error.kind));
      box.appendChild(cell('Stage', t.error.stage || 'router'));
      box.appendChild(cell('Cached', 'nothing'));
      err.hidden = false;
      err.textContent = t.error.detail;
      return;
    }
    $('d-cost-title').textContent = 'WHAT THIS CALL WILL COST';
    box.appendChild(cell('Tokens', 'pending'));
    box.appendChild(cell('Model', t.config.llm_model));
  }

  function renderIdentity(t) {
    var dl = $('d-ident');
    dl.textContent = '';
    var rows = [['Request', t.id], ['Arrived', new Date(t.arrived_at * 1000).toISOString()]];
    var e = t.embedding;
    if (e && t.config.semantic_cache) {
      rows.push(['Embedding', [e.provider, e.checkpoint || e.canonical_name, e.dimension ? e.dimension + '-d' : null].filter(Boolean).join(' · ')]);
    }
    if (t.llm) rows.push(['Provider', [t.llm.provider, t.llm.endpoint, t.llm.model].join(' · ')]);
    rows.push(['Engine', t.config_key]);
    rows.forEach(function (r) {
      dl.appendChild(el('dt', null, r[0]));
      dl.appendChild(el('dd', null, r[1]));
    });
  }

  // -- controls ----------------------------------------------------------------

  $('follow').addEventListener('click', function () {
    setFollow(!state.follow);
    if (state.follow && state.order.length) {
      state.selected = state.order[state.order.length - 1];
      var t = state.traces.get(state.selected);
      if (t) play(t, true);
      render();
    }
  });

  $('clear-history').addEventListener('click', function () {
    fetch('/console/traces/clear', { method: 'POST' }).then(function () {
      resetHistory();
      render();
    });
  });

  // Keep in-flight elapsed times moving between polls.
  setInterval(function () {
    var t = state.selected ? state.traces.get(state.selected) : null;
    if (t && t.status === 'in_flight') {
      $('d-total').textContent = fmtMs(elapsedMs(t)) + ' and counting';
      renderSpans(t);
      $('story').lastChild.textContent = storyText(t);
    }
  }, 100);

  setConnected(false);
  render();
  poll();
})();
