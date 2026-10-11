"""
Levy Console: a `console` section of the API beside the delivered
endpoints, plus the observability page that shows each console request live.

- `options`  -- the closed lists of real settings a console request may choose.
- `pool`     -- the console's own engines, one per resolved configuration.
- `tracing`  -- per-request trace capture and the bounded in-memory feed.
- `routes`   -- the `/console/*` endpoints and the page.

The delivered `/v1/chat/completions` and `/admin/*` endpoints are untouched:
the console never shares their engines or caches.
"""
