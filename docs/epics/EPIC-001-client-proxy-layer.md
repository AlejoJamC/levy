# EPIC-001: Client/Proxy Layer (Request Interception & Routing)

> **Status: historical planning document.** This epic was written before the
> layer was built and is kept for provenance, not as a description of the
> released system. The client/proxy layer has shipped — see
> [`levy/api/`](../../levy/api/) and
> [docs/ARCHITECTURE.md](../ARCHITECTURE.md) for what was actually built. Where
> this document and the code differ, the code and the architecture document are
> authoritative; for research scope, the frozen
> [Project_Proposal.md](../Project_Proposal.md) and
> [Specification_and_Design_Report.md](../Specification_and_Design_Report.md)
> win.

**Epic ID**: EPIC-001  
**Component**: Component 1 of 5  
**Status**: 📋 Planning (superseded — shipped as `levy/api/`)  
**Owner**: Alejandro Mantilla  
**Timeline**: Weeks 4-5 (Phase Two)  
**Priority**: P0 (Critical Path)

---

## 🎯 Epic Goal

Build the foundational request interception and routing layer that enables experimental comparison of different caching strategies (no cache, exact-match cache, semantic cache) with full observability and provider independence.

---

## 📊 Business Value (Academic Context)

As a researcher validating semantic caching effectiveness, I need a transparent interception point between the application and the LLM provider so that I can:

1. **Capture all requests** for experimental reproducibility
2. **Route queries** through different caching strategies systematically
3. **Measure latency** from request arrival to response delivery
4. **Compare strategies** fairly using identical request sequences
5. **Maintain provider independence** for generalizability

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT/PROXY LAYER                       │
│                                                             │
│  ┌──────────────┐                                          │
│  │  REST API    │  ← PHASE 1 (Week 4-5)                    │
│  │  /api/query  │                                          │
│  └──────┬───────┘                                          │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────┐         │
│  │         REQUEST LOGGER & VALIDATOR           │         │
│  │  - Assign request_id                         │         │
│  │  - Timestamp arrival                         │         │
│  │  - Validate payload                          │         │
│  └──────────────┬───────────────────────────────┘         │
│                 │                                          │
│                 ▼                                          │
│  ┌──────────────────────────────────────────────┐         │
│  │           ROUTING LOGIC                      │         │
│  │  if cache_strategy == "none":                │         │
│  │      → Component 5 (LLM Backend)             │         │
│  │  elif cache_strategy == "exact":             │         │
│  │      → Component 2 (Exact Cache)             │         │
│  │  elif cache_strategy == "semantic":          │         │
│  │      → Component 3 (Semantic Cache)          │         │
│  └──────────────┬───────────────────────────────┘         │
│                 │                                          │
│                 ▼                                          │
│  ┌──────────────────────────────────────────────┐         │
│  │         RESPONSE FORMATTER                   │         │
│  │  - Add metadata (source, latency, cost)      │         │
│  │  - Timestamp completion                      │         │
│  │  - Log result                                │         │
│  └──────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Scope Definition

### ✅ In Scope

- REST API endpoint for query submission
- Request validation and structured payload
- Request/response logging for experiments
- Routing logic to different caching strategies
- Latency measurement (end-to-end)
- Provider-agnostic response format
- Configuration mechanism for cache strategy selection

### ❌ Out of Scope

- Caching logic implementation (Component 2 & 3)
- Embedding computation (Component 3)
- Vector database operations (Component 4)
- LLM API calls (Component 5)
- Authentication/authorization (not needed for research prototype)
- Rate limiting (handled by LLM provider)

---

## 🔑 Key Design Decisions

### Decision 1: Interface Strategy (Prioritized Roadmap)

**Phase 1 (Weeks 4-5): REST API** ✅ APPROVED
- **Why**: Universal, easy to test, curl/Postman friendly
- **Framework**: FastAPI (async, auto OpenAPI docs)
- **Enables**: All basic experiments, external tools integration

**Phase 2 (Week 6): Python SDK**
- **Why**: Better DX for notebook-driven research
- **Implementation**: Wrapper around REST API
- **Enables**: `levy.generate(prompt)` instead of HTTP calls

**Phase 3 (Week 7-8, if time): MCP Protocol**
- **Why**: Enable AI agents (Claude, GPT) to use Levy
- **Implementation**: MCP server wrapping Levy
- **Enables**: AIX/AX use cases, agent-to-agent scenarios

**Phase 4 (Post-MSc): gRPC**
- **Why**: Production performance for commercial deployment
- **Status**: Future work, not critical for research validation

### Decision 2: Request Structure

**Structured Entity with Metadata** ✅ APPROVED

```python
{
    "prompt": "What is semantic caching?",  # User's raw query (string)
    "cache_strategy": "semantic",            # "none" | "exact" | "semantic"
    "similarity_threshold": 0.85,            # For semantic cache
    "llm_config": {                          # Optional LLM parameters
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 500
    },
    "metadata": {                            # Optional experimental context
        "user_id": "researcher_01",
        "session_id": "exp_2024_12_15",
        "workload_type": "faq"
    }
}
```

**Why structured?**
- Reproducibility: Track which strategy was used per request
- Experimental control: A/B test different thresholds
- Future-proof: Easy to add new parameters without breaking API

### Decision 3: Configuration Evolution Path

**Week 4-5: Simple Function Parameters**
```python
result = levy.generate(
    prompt="What is semantic caching?",
    cache_strategy="semantic",
    similarity_threshold=0.85
)
```

**Week 6: Configuration Object (Recommended for Experiments)**
```python
config = LevyConfig(
    cache_strategy="semantic",
    similarity_threshold=0.85,
    llm_provider="openai",
    embedding_model="modernbert"
)
levy = LevyEngine(config)
result = levy.generate("What is semantic caching?")
```

**Post-MSc: Multi-Layer Configuration**
```
Priority: Runtime params > Config file (levy.yaml) > Environment vars (.env)
```

**Rationale**: Start simple for MVP, evolve as experiments become more complex. Configuration object is the proper pattern because:
- Centralized: One place to see all experiment settings
- Reusable: Load config from YAML for reproducible experiments
- Typed: Pydantic validation prevents configuration errors

---

## 🎯 Success Criteria (Acceptance Criteria)

This Epic is **DONE** when:

✅ REST API endpoint `/api/query` accepts POST requests with structured JSON payload  
✅ Every request is assigned a unique `request_id` and logged with timestamp  
✅ Routing logic correctly directs requests based on `cache_strategy` parameter  
✅ Response format includes: `content`, `source` (llm/exact_cache/semantic_cache), `latency_ms`, `estimated_cost`  
✅ Swapping LLM provider (OpenAI → Ollama) requires changing only Component 5, not Component 1  
✅ Manual testing via curl demonstrates: baseline (no cache), exact-match caching, and semantic caching  
✅ Logs contain enough information to replay any experiment  

---

## 📦 Deliverables

1. **Code**:
   - `levy/proxy/api.py` - FastAPI REST endpoint
   - `levy/proxy/router.py` - Routing logic
   - `levy/proxy/logger.py` - Request/response logging
   - `levy/models.py` - Pydantic models for LevyRequest/LevyResponse

2. **Documentation**:
   - `docs/API_REFERENCE.md` - REST API specification
   - `examples/curl_examples.sh` - Sample requests

3. **Tests**:
   - `tests/test_proxy_layer.py` - Unit tests for routing logic

---

## 🔗 Dependencies

### Upstream (Must Complete Before Starting)
- ✅ Python environment setup (conda)
- ✅ Project structure finalized
- ⏳ Data models defined (LevyRequest, LevyResponse, LevyConfig)

### Downstream (Depends on This Epic)
- Component 2: Exact-Match Cache (needs routing from Component 1)
- Component 3: Semantic Cache (needs routing from Component 1)
- Component 5: LLM Backend (needs request format from Component 1)

---

## 🚧 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| REST API framework choice delays implementation | Medium | Use FastAPI (familiar, auto docs, async) |
| Configuration becomes too complex | High | Start with simple parameters, evolve incrementally |
| Logging overhead impacts latency | Low | Use async logging, minimal overhead design |
| Request format changes break downstream components | High | Define strict Pydantic schemas early, version API |

---

## 📝 Open Questions (To Resolve Before Implementation)

1. **Where do logs go?**
   - Option A: SQLite database (queryable, structured)
   - Option B: JSON files (simple, git-friendly for small experiments)
   - Option C: Stdout only (debugging, parse later)
   - **Recommendation**: Start with JSON files, evolve to SQLite if experiments scale

2. **How granular should latency tracking be?**
   - Option A: Total end-to-end only
   - Option B: Breakdown per stage (routing, cache lookup, LLM call, embedding)
   - **Recommendation**: Start with end-to-end, add breakdown if needed for analysis

3. **Should we support batch requests?**
   - Not critical for Phase 1 (single query experiments)
   - Consider for Phase 2 if workload replay needs it

---

## 🛠️ Technical Stack

- **Web Framework**: FastAPI 0.104+
- **Validation**: Pydantic v2
- **Logging**: Python `logging` + JSON formatter
- **Testing**: pytest
- **Documentation**: Auto-generated OpenAPI (FastAPI built-in)

---

## 📅 Timeline (Estimated)

- **Week 4.1**: Define data models (LevyRequest, LevyResponse, LevyConfig)
- **Week 4.2**: Implement REST API endpoint skeleton
- **Week 4.3**: Add routing logic (3 strategies)
- **Week 4.4**: Implement logging & response formatting
- **Week 5.1**: Integration testing with mock LLM
- **Week 5.2**: Documentation & curl examples
- **Week 5.3**: Buffer for issues

**Total Estimated Effort**: 8-10 days

---

## 🔄 Next Steps

1. **Create Feature-001**: Define Pydantic data models
2. **Create Feature-002**: Implement FastAPI skeleton
3. **Create Feature-003**: Build routing logic
4. **Create Feature-004**: Add logging layer
5. **Create Spike-001**: Research MCP integration (for Phase 3)

---

## 📚 References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- Pydantic v2 Guide: https://docs.pydantic.dev/latest/
- REST API Best Practices: https://restfulapi.net/
- Model Context Protocol (MCP): https://modelcontextprotocol.io/

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Contact**: alejojamc7@gmail.com
