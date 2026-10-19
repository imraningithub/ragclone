# Jan vs. clone: retrieval comparison

**Date:** 2026-10-19
**Jan:** 0.8.6. Its embedding model is `all-MiniLM-L6-v2` (`sentence-transformer-mini`, 384 dimensions).
**Clone:** `ragclone/`. It uses OpenAI `text-embedding-3-small` and `gpt-4.1-mini`.
**Documents:** `samples/` (`employee_handbook.pdf`, `r7_product_spec.docx`, `sales_and_prices.xlsx`). In Jan they were attached with the parse mode set to "Ingest as embeddings".

**Questions:** 8 (see `scripts/compare_clone.py`). They cover all three files, include one two-part question that needs two spreadsheet sheets, and one off-topic question.

## How Jan was measured (and how much to trust it)
- **Jan's index is real.** Jan ingested the files itself. I copied its DB, `data/db/attachments_<thread>.db`, to `compare/jan_snapshot/`.
- **Jan's answers aren't available.** Its local model, Jan-v3.5-4B, failed to load on this machine: `llama.cpp ran out of memory` (Vulkan `ErrorOutOfDeviceMemory`). So only retrieval is compared, not Jan's answers.
- **Query embeddings came from Jan's own model,** through Jan's Local API Server. Re-embedding Jan's stored chunks through that API reproduced the stored vectors exactly (cosine **1.0, 1.0, 1.0**).
- **Ranking:** for Jan's ranking I applied its settings, which I saw in its UI: linear cosine search, Top-K 3, threshold 0.3. Jan's DB has no `vec0` table, so it searched linearly here.
- **Inference, not observed:** I didn't see Jan run its own query. The ranking assumes Jan embeds the raw question the same way.

## Result: was the answer-bearing chunk retrieved?

| # | Question | Jan | Clone + MiniLM (k3, 0.3) | Clone + OpenAI (k3, 0.3) | Clone + OpenAI (k5, 0.2, tuned) |
|---|---|---|---|---|---|
| 1 | Vacation days + carry-over deadline | ✅ | ✅ | ✅ | ✅ |
| 2 | Work from home on Fridays? | ✅ | ✅ | ✅ | ✅ |
| 3 | Hotel limit in big cities | ✅ | ✅ | ✅ | ✅ |
| 4 | Minimum password length | ✅ | ✅ | ✅ | ✅ |
| 5 | Full charge time of R7 | ✅ | ✅ | ✅ | ✅ |
| 6 | Battery warranty | ✅ | ✅ | ✅ | ✅ |
| 7 | West Q4 units **and** R7 price (2 sheets) | ✅ | ❌ | ❌ | ✅ |
| 8 | Capital of Australia (off-topic, expect none) | ✅ none | ✅ none | ✅ none | ✅ none |
| | **Total** | **8/8** | 7/8 | 7/8 | **8/8** |

The full hits and scores are in `jan_vs_clone.json`.

## Why Jan wins Q7 at k=3: chunking, not embeddings
The clone with Jan's own MiniLM model misses Q7 in the same way the OpenAI version does. So the difference comes from chunking:

| | Jan | Clone |
|---|---|---|
| Chunk cutting | Fixed 512-char windows. Cuts land mid-word ("approved in advance by the l"). | Snaps to the paragraph, then sentence, then line, then space within the last 20% of the window. |
| PDF text | Puts a blank line between every wrapped line. | Keeps single line breaks. |
| XLSX rows | Tab-separated, header row once per sheet. | Before the hybrid update: `Column: value; …` on every row (see Update below). |
| XLSX chunks | **2**: the West Q4 row and the price list share chunk #1. | **5**: they are in separate chunks. |
| DOCX tables | One cell per line. | `cell \| cell` per row. |
| Bugs seen | "°C" from the DOCX is stored as "�C". | none seen |

There is a trade-off. Jan's compact table format keeps small tables in few chunks, which helps multi-part questions. But in Jan's later spreadsheet chunks, rows lose their column names: chunk #1 holds rows like `West	Q4 2025	700	2.1	R7` with no header. The clone's format makes every chunk self-describing, at the cost of more chunks. Raising the clone's Top-K to 5 closes the gap.

## Answer quality (clone only)
- **At k=3 the clone answered Q7 wrong:** it said the R7 costs 3,000 EUR. Without the price-list chunk, the model computed revenue ÷ units (2.1M EUR ÷ 700) and presented that as the price, even with the "don't guess" instruction.
- **At k=5 it answered correctly:** 54,000 EUR, citing the price-list chunk.
- **Off-topic question:** no excerpts were retrieved in any variant, and the model answered from general knowledge.

## What else this confirmed about Jan (runtime, not just static analysis)
- **Every default in Jan's Attachments UI matches the strings in its binary:** Auto parse, 0.75 auto-inline ratio, 100 MB, Top-K 3, threshold 0.3, chunk 512, overlap 64, Auto search mode.
- **Storage is one SQLite DB per collection,** at `data/db/attachments_<threadId>.db`. Its tables are `files` and `chunks(id, text, embedding BLOB, file_id, chunk_file_order)`. The clone uses one shared DB instead.
- **Jan logs each parse:** `tauri_plugin_rag::commands: Parsing document: <path> (type: pdf)`.
- **The auto-inline ratio applies to local models.** The UI help text says it is a "fraction of the model context … on local models".
- **Still unknown:** how Jan inserts retrieved chunks into the prompt. Its chat model couldn't run here, so this needs a working Jan model, either on a CPU backend or a remote provider.

## Update: hybrid spreadsheet format (implemented)
**The new format:** spreadsheets are now parsed into compact tables, a `# Sheet: name` line, then a `col | col` header row, then `v | v` rows. The chunker packs **whole rows** into each chunk and **repeats the sheet line and header row at the top of every chunk**. It never splits a row and needs no overlap for tables. Prose chunking is unchanged.

| Spreadsheet chunks | Old clone | Jan | Hybrid (new clone) |
|---|---|---|---|
| Count for `sales_and_prices.xlsx` | 5 | 2 | **3** |
| Every chunk has its column names | yes | no | **yes** |
| Rows split across chunks | yes (fixed later) | yes (cut mid-row) | **never** |

Rerun results (same 8 questions):

| | Jan | Clone + MiniLM (k3, 0.3) | Clone + OpenAI (k3, 0.3) | Clone + OpenAI (k5, 0.2) |
|---|---|---|---|---|
| Before hybrid | 8/8 | 7/8 | 7/8 | 8/8 |
| **After hybrid** | 8/8 | 7/8 | **8/8** | 8/8 |

- **OpenAI at Jan's settings now answers Q7 correctly:** "700 units … 54,000 EUR". It cites the sales chunk and the price-list chunk.
- **Clone + MiniLM still misses Q7 at k=3, but for a different reason.** It now retrieves the price-list chunk, but ranks a DOCX spec chunk (0.446) above the West sales chunk. This is a close ranking call by the small MiniLM model, not a chunking defect.
- **Threshold eval:** 0.2 is still the best setting (21/21 hits, 0/5 off-topic).
- **Tests:** 60 pass.

## Possible next steps for the clone
1. **Multi-part questions:** keep Top-K 5 as the default, or retrieve per sub-question.
2. **Local-model mode:** if you switch the clone to Jan's MiniLM, consider Top-K 5 there too.
