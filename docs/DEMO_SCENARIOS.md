# Demo scenarios

`POST /console/chat` in `http://localhost:8000/docs`, with `http://localhost:8000/console` beside it.
Start from a freshly started Levy (empty caches).

Fields that stay the same in every scenario:

| Field | Value |
|---|---|
| embedding_provider | `sentence-transformers` |
| embedding_model | `all-MiniLM-L6-v2` |
| llm_provider | `anthropic` |
| llm_model | `claude-haiku-4-5-20251001` |
| exact_cache | `true` |
| semantic_cache | `true` |
| vector_index_backend | `faiss` |
| cache_store | `memory` |

## UC1 · Cold miss

prompt

```
How do I reset my password?
```

threshold

```
0.70
```

Expected: MISS, sent to Anthropic, answer stored. The semantic index was empty.

## UC2 · Exact hit

prompt

```
How do I reset my password?
```

threshold

```
0.70
```

Expected: exact HIT, `x-cache-similarity: 1.0`, answer written for the same text.

## UC3 · Semantic hit

prompt

```
How can I reset my password?
```

threshold

```
0.70
```

Expected: semantic HIT, similarity 0.8857 ≥ τ 0.70.

## UC4 · Correct rejection

prompt

```
What is the refund window for annual plans?
```

threshold

```
0.70
```

Expected: MISS, nearest stored prompt "How do I reset my password?" at 0.4360 < τ 0.70.

## UC5 · False positive

prompt

```
What is the refund window for monthly plans?
```

threshold

```
0.70
```

Expected: semantic HIT, 0.7046 ≥ τ 0.70. It receives the annual-plan answer.

## UC6 · Same prompt at a stricter threshold

Each threshold has its own cache, so the first request stores the annual-plan prompt at τ 0.75.

prompt

```
What is the refund window for annual plans?
```

threshold

```
0.75
```

Expected: MISS, stored.

prompt

```
What is the refund window for monthly plans?
```

Expected: MISS, 0.7046 < τ 0.75 (τ 0.75 = cosine 0.944).

## UC7 · Provider failure

Stop Ollama:

```bash
osascript -e 'quit app "Ollama"'
```

| Field | Value |
|---|---|
| llm_provider | `ollama` |
| llm_model | `qwen3.8:latest` |

prompt

```
Do you offer student discounts?
```

threshold

```
0.70
```

Expected: HTTP 500 `provider_error`, nothing cached.

Start Ollama again:

```bash
open -a Ollama
```
