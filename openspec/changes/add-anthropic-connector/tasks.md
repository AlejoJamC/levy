# Tasks: add-anthropic-connector

## 1. Dependency & config

- [ ] 1.1 Add `anthropic` SDK to `environment.yml` (pip section) and `pyproject.toml` dependencies; install into the `levy` conda env and verify import.
- [ ] 1.2 Extend `LevyConfig`: `anthropic_api_key` (from `ANTHROPIC_API_KEY`), `anthropic_model` (default `claude-opus-4-8`), `anthropic_max_retries` (default 2), `anthropic_budget_cap_usd` (default 200.0), `anthropic_input_price_per_mtok` / `anthropic_output_price_per_mtok` (defaults matching the default model). Add `ANTHROPIC_API_KEY` placeholder line to `.env.example`.

## 2. Connector

- [ ] 2.1 Implement `BudgetExceededError` + budget accumulator (request count, estimated cost from usage × prices, `check()` before each call, exposed `request_count` / `estimated_cost_usd`).
- [ ] 2.2 Implement `AnthropicLLMClient(LLMClient)` in `levy/llm_client.py`: sync `anthropic.Anthropic` client with `max_retries` from config and optional injectable `http_client`; `generate()` sends a single-turn `messages.create` (model, max_tokens, prompt), checks `stop_reason` (refusal → clear error), builds `LLMResponse` with `token_usage = input + output` and metadata split (input/output tokens, model, stop_reason); missing API key fails at construction naming the variable.
- [ ] 2.3 Wire the `"anthropic"` branch into the engine's provider factory; mock/OpenAI/Ollama untouched.

## 3. Tests (offline, mocked transport)

- [ ] 3.1 Transport fixture: `anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler))` with scripted wire responses (success + usage block, 429-then-success, 400, refusal stop_reason).
- [ ] 3.2 Tests: success populates text/token_usage/metadata; retry-then-success returns transparently; 400 propagates typed; refusal raises and nothing is cached; budget guard halts at the cap with no request sent and exposes count/estimate; missing-key construction error; engine end-to-end with `llm_provider="anthropic"` over mock transport.
- [ ] 3.3 Full gated suite green in the conda env: `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` — no new pragmas; connector fully inside the denominator.

## 4. Manual smoke path (not in CI)

- [ ] 4.1 Document (README section) a one-command real-API smoke check using a real `ANTHROPIC_API_KEY` from `.env`, for the author to run once before experiment-era use; verify it is excluded from the offline suite.

## 5. Docs & sync

- [ ] 5.1 Update README.md (anthropic provider config, budget guard, smoke check) and CLAUDE.md (architecture map entry, mark known-gap #2 resolved, note the sync-now/async-at-router decision and the retired-model drift flag on the frozen S&D example).
- [ ] 5.2 `openspec validate --all` passes; keep Linear LEV-6 in sync (reference this change; tick acceptance criteria as delivered; record the sync/async decision the issue asked this change to settle).
