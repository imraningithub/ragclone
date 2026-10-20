# OpenAI API pricing and token-usage reporting (gpt-4.1, gpt-4.1-mini, text-embedding-3-small)

Retrieved: 2026-10-20. All sources are OpenAI first-party pages, plus the official `openai/openai-python` SDK on GitHub (`main` branch). `https://platform.openai.com/docs/pricing` now returns a 301 redirect to `https://developers.openai.com/api/docs/pricing`, so the `developers.openai.com` URLs below are the canonical ones.

## 1. Summary

At Standard (non-batch, default-tier) pricing, gpt-4.1 costs $2.00 per 1M input tokens, $0.50 per 1M cached input tokens, and $8.00 per 1M output tokens. gpt-4.1-mini costs $0.40, $0.10, and $1.60 for the same three. text-embedding-3-small costs $0.02 per 1M input tokens. It has no cached-input or output price. I found no Batch price for it on the pricing page or on its model page. Chat Completions reports usage as `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`, and `usage.prompt_tokens_details.cached_tokens`. `cached_tokens` counts the prompt tokens that were cache hits, so it is a subset of `prompt_tokens`. When streaming with `stream_options={"include_usage": true}`, usage arrives only in one extra final chunk whose `choices` is an empty array, and that chunk may never arrive if the stream is interrupted or cancelled. Embeddings report `usage.prompt_tokens` and `usage.total_tokens`. Prompt caching is on by default and needs no code changes. For gpt-4.1-era models the current guide no longer gives a fixed minimum. The older 1,024-token minimum appears only in an archived cookbook.

## 2. Price table (USD per 1M tokens, Standard tier)

| Model | Input | Cached input | Output | Source URL |
|---|---|---|---|---|
| gpt-4.1 | $2.00 | $0.50 | $8.00 | https://developers.openai.com/api/docs/pricing (Standard table); confirmed at https://developers.openai.com/api/docs/models/gpt-4.1 |
| gpt-4.1-mini | $0.40 | $0.10 | $1.60 | https://developers.openai.com/api/docs/pricing (Standard table); confirmed at https://developers.openai.com/api/docs/models/gpt-4.1-mini |
| text-embedding-3-small | $0.02 | n/a ("-") | n/a ("-") | https://developers.openai.com/api/docs/pricing (Specialized models > Standard table, "Embedding" row); confirmed at https://developers.openai.com/api/docs/models/text-embedding-3-small |

Other tiers, for reference only. The app uses Standard.

| Model | Tier | Input | Cached input | Output | Source |
|---|---|---|---|---|---|
| gpt-4.1 | Batch | $1.00 | - | $4.00 | https://developers.openai.com/api/docs/pricing (Batch table) |
| gpt-4.1-mini | Batch | $0.20 | - | $0.80 | same |
| gpt-4.1 | Fast (formerly "Priority"; the page says it was renamed on 2026-07-30) | $3.50 | $0.875 | $14.00 | same (Fast table) |
| gpt-4.1-mini | Fast | $0.70 | $0.175 | $2.80 | same |
| text-embedding-3-small | Batch | **Not listed** | - | - | The pricing page has no Batch row for it. The model page says the Batch endpoint is supported but gives no batch price. A Batch price for this model is **UNVERIFIED**. |

Neither Flex table lists gpt-4.1 or gpt-4.1-mini (source: pricing page).

## 3. Usage-reporting facts

### Chat Completions (non-streamed)
- `usage.prompt_tokens` is defined as "Number of tokens in the prompt." Sources: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events (shared `CompletionUsage` schema) and https://github.com/openai/openai-python/blob/main/src/openai/types/completion_usage.py
- `usage.completion_tokens` is defined as "Number of tokens in the generated completion." It includes reasoning tokens and rejected prediction tokens: the SDK docstring for `rejected_prediction_tokens` says these "are still counted in the total completion tokens for purposes of billing". Source: same SDK file.
- `usage.total_tokens` is defined as "Total number of tokens used in the request (prompt + completion)." Source: same.
- `usage.prompt_tokens_details.cached_tokens` is defined as "Cached tokens present in the prompt." `prompt_tokens_details` is `Optional` in the SDK, and so is `cached_tokens`, so either can be `None`. The SDK also has `prompt_tokens_details.cache_write_tokens`, described as "The unadjusted number of prompt tokens written to cache." Source: https://github.com/openai/openai-python/blob/main/src/openai/types/completion_usage.py
- **Is `cached_tokens` a subset of `prompt_tokens`?** Yes, though no page states it in exactly those words. The archived cookbook says the field indicates "how many of the prompt tokens were a cache hit," and its example responses show `prompt_tokens: 1136` with `cached_tokens: 1024`, and `prompt_tokens: 1548` with `cached_tokens: 1280`. Source: https://developers.openai.com/cookbook/examples/prompt_caching101 (marked "archived"). The current prompt-caching guide's cost formula computes `input_tokens - cached_tokens - cache_write_tokens` as ordinary input, which is consistent with this. That guide uses the Responses API names (`input_tokens`), not Chat Completions names. Source: https://developers.openai.com/api/docs/guides/prompt-caching
- I could not fetch the Chat Completions create reference page (https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). It returned HTTP 404 to WebFetch on 2026-10-20. The field definitions above come from the streaming-events reference page and the SDK, both generated from the same OpenAPI spec.

### Chat Completions (streamed, `stream_options={"include_usage": true}`)
- `include_usage`: "If set, an additional chunk will be streamed before the `data: [DONE]` message. The `usage` field on this chunk shows the token usage statistics for the entire request, and the `choices` field will always be an empty array. All other chunks will also include a `usage` field, but with a null value." Sources: https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_stream_options_param.py and the reference page text (via search snippet) at https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
- `ChatCompletionChunk.choices`: "Can also be empty for the last chunk if you set `stream_options: {"include_usage": true}`." Sources: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events and https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_chunk.py
- `ChatCompletionChunk.usage`: present only with `include_usage`, and "contains a null value **except for the last chunk**."
- **Interrupted streams:** "If the stream is interrupted or cancelled, you may not receive the final usage chunk which contains the total token usage for the request." Sources: same two pages.
- `stream_options` should only be set when `stream: true`. Source: `chat_completion_stream_options_param.py` docstring.

### Embeddings
- `usage.prompt_tokens` is "The number of tokens used by the prompt." `usage.total_tokens` is "The total number of tokens used by the request." Sources: https://developers.openai.com/api/reference/resources/embeddings/methods/create and https://github.com/openai/openai-python/blob/main/src/openai/types/create_embedding_response.py. Both fields are required `int`s. There is no cached or output count.

### Prompt caching
- **Automatic:** "Prompt caching is enabled by default for supported OpenAI models." Implicit caching needs no code changes. Source: https://developers.openai.com/api/docs/guides/prompt-caching. The guide does not list every supported model. It names `gpt-4.1` only in a footnote on models that support extended (`24h`) retention. **gpt-4.1-mini is not named.** That gpt-4.1-mini gets automatic caching is consistent with the $0.10 cached-input price on the pricing page, but the guide does not say so explicitly (**UNVERIFIED** from the guide alone).
- **Minimum length (current guide):** "The minimum cacheable prompt length is 1,024 tokens for GPT-5.6 and later and varies by request settings for earlier models." For earlier models, the minimum depends on "tools, images, output schemas, reasoning effort, and verbosity." Source: prompt-caching guide. gpt-4.1 and gpt-4.1-mini predate GPT-5.6, so the current guide gives **no fixed minimum** for them.
- **Minimum length (archived cookbook):** "Prompt caching automatically activates for prompts longer than 1024 tokens", and for requests under 1024 tokens `cached_tokens` is zero. Source: https://developers.openai.com/cookbook/examples/prompt_caching101 (archived, and its examples use gpt-4o). Treat 1,024 as the historical floor, not a current guarantee for gpt-4.1.
- **Granularity:** "Reported `cached_tokens` is calculated by subtracting the hidden system tokens from the last matched breakpoint, then rounding down to the nearest multiple of 128." Source: prompt-caching guide.
- **No extra write charge for earlier models:** the guide's table says "No additional cache-write charge" for pre-GPT-5.6 models. The 1.25x cache-write rate applies only to GPT-5.6 and later. Source: prompt-caching guide.
- **Retention (earlier models):** `prompt_cache_retention` is `"in_memory"` (about 5 to 10 minutes of inactivity, up to 1 hour) or `"24h"`. Organizations without Zero Data Retention default to `24h`. Source: prompt-caching guide.

## 4. Notes for implementers

Cost per Chat Completions call (Standard tier, gpt-4.1 / gpt-4.1-mini):

```
prompt    = usage.prompt_tokens
cached    = (usage.prompt_tokens_details.cached_tokens if details present else 0) or 0
completion= usage.completion_tokens

cost_usd = ( (prompt - cached) * INPUT_PER_1M
           + cached            * CACHED_INPUT_PER_1M
           + completion        * OUTPUT_PER_1M ) / 1_000_000
```

Rates: gpt-4.1 is 2.00 / 0.50 / 8.00. gpt-4.1-mini is 0.40 / 0.10 / 1.60.

Embeddings: `cost_usd = usage.prompt_tokens * 0.02 / 1_000_000` for text-embedding-3-small.

Example: gpt-4.1-mini with prompt=1,548, cached=1,280, completion=200 costs (268 x 0.40 + 1,280 x 0.10 + 200 x 1.60) / 1e6 = $0.0005552.

Caveats:
- **Null handling:** `prompt_tokens_details` and `cached_tokens` are `Optional` in the SDK. Treat `None` as 0. Community reports describe the details object as missing in some gpt-4.1 setups, but those are forum reports and not authoritative.
- **Streaming:** usage is only in the final chunk, which has `choices == []`. Code that reads `chunk.choices[0]` without a guard will raise `IndexError` on that chunk. If a stream is interrupted, cancelled, or abandoned by the client, there will be no usage at all. Decide on a fallback: record the call as "usage unknown", or estimate with a tokenizer and mark it as an estimate. Don't silently record $0.
- **cache_write_tokens:** the SDK now exposes `prompt_tokens_details.cache_write_tokens`. For pre-GPT-5.6 models the guide says there is no extra write charge, so the formula above stays correct for gpt-4.1 / gpt-4.1-mini. If the app moves to GPT-5.6+, add a third input bucket at 1.25x the input rate and subtract it from ordinary input, as the guide's formula does.
- **Reasoning and predicted-output tokens** are already inside `completion_tokens` and billed at the output rate. Don't add them again.
- **Service tier:** these prices assume Standard. If a request sets or inherits `service_tier` "flex" or "fast"/"priority" (the response reports `service_tier`), different rates apply. Read `response.service_tier` if it matters.
- **Long context:** the pricing page labels the Standard table "short context ≤272K tokens". I did not check whether gpt-4.1 / gpt-4.1-mini have separate long-context (>272K) rates (**UNVERIFIED**). Keep prompts in the RAG app well below that.
- **Prices change.** Neither page shows a "last updated" date. Keep the rates in config, not hard-coded, and re-check the pricing URL periodically.
- **Unfetchable page:** https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create returned 404 to the fetch tool. Its `include_usage` text above comes from a search-result snippet and matches the SDK docstring word for word.
