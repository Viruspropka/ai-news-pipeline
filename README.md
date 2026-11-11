# ai-news-pipeline

A small, production-shaped pipeline that turns public AI news feeds into a short daily brief. It ingests RSS/Atom feeds, drops duplicates, has an LLM score every item, has a second, stricter LLM pass reject hype and filler, and composes a brief in the language you configure. The brief goes to pluggable sinks (console, file, optionally Telegram).

Every model call goes through one adapter. A deterministic mock backend makes the whole thing, including the demo, the tests and the eval, run offline without API keys.

## What this demonstrates

LLM output treated as untrusted input: typed JSON validated against a schema on every backend, links taken from the database instead of the model, article text passed as data with an explicit "do not follow instructions in it" rule. Failure isolation at every level: one bad feed, one malformed model answer or one broken sink never stops the run, while a used-up model plan stops it cleanly instead of failing every item. A vendor-neutral model layer (Anthropic SDK, any CLI, or a mock) chosen by config. And a small labelled eval that reports the number that matters for a "no hype" brief: how many non-pass items the gate lets through. It is a clean rewrite of the core of a pipeline that runs on a schedule in production; no data, keys or git history came with it.

## Stages

```mermaid
flowchart LR
    F[sources.yaml<br/>RSS / Atom feeds] --> I[ingest<br/>canonical URL + content hash]
    I --> D[dedupe<br/>in batch + in store]
    D --> S[score<br/>relevance / novelty /<br/>actionability / category]
    S --> V[verify top K<br/>pass / hype / dubious / thin]
    V --> B[brief<br/>window, verdict gate,<br/>cooldown, diversity caps]
    B --> C[compose<br/>output language,<br/>banned-phrase guard]
    C --> K{sinks}
    K --> K1[console]
    K --> K2[file .md + .jsonl]
    K --> K3[telegram, optional]
    L[(LLM adapter<br/>mock / anthropic / command)] -.-> S & V & C
    Q[quota guard] -.-> L
```

## Architecture in ten lines

1. `sources/`: adapters turn a source entry into `RawArticle`s; `registry.py` validates `sources.yaml` strictly and runs ingest in weight order, so the primary source wins a dedupe.
2. `stages/dedupe.py`: pure function, drops in-batch copies by canonical URL (tracking parameters, `www.`, fragments removed) or content hash.
3. `store.py`: SQLite; `UNIQUE(url_canonical)` plus a content-hash check catch copies across runs.
4. `stages/score.py`: one typed-JSON call per item; `compute_composite` is the single ranking formula (weights in config, times source weight).
5. `stages/verify.py`: a stricter pass on the top K only, so the expensive model never sees the long tail.
6. `stages/brief.py`: selection is plain code; the model writes headline and blurb per item; unknown ids and items that still use banned phrases after one retry are dropped.
7. `llm/`: `LLMBackend.complete_json()` is the only call stages make; `validate()` checks every answer against the schema, whatever the backend.
8. `quota.py`: `QuotaExhausted` writes a stop file; later runs exit with code 75 until the reset time.
9. `sinks/`: `deliver()` sends to each sink and records per-sink results; one failure never blocks the others.
10. `pipeline.py` wires the stages; `cli.py` exposes `run`, `demo`, `eval` and `sources`.

## Quickstart (offline, no keys)

Python 3.11 or newer.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
ai-news-pipeline demo          # synthetic feeds + mock model, writes to a temp dir
pytest                         # 155 tests, about 3 s
ai-news-pipeline eval          # labelled cases through score + verify
```

Docker (same demo, not built yet, see limits): `docker build -t ai-news-pipeline . && docker run --rm ai-news-pipeline`.

### Demo output (shortened)

```text
Run 1 at 2026-03-11 08:00 UTC: three synthetic feeds and one broken source
------------------------------------------------------------------------
AI news brief - 2026-03-11

NEW MODELS & LLMS
- Introducing Nimbus-7B, an open-weight reasoning model
  We are releasing Nimbus-7B, an open-weight language model for reasoning tasks.
  Source: Example Vendor Blog - https://vendor.example.com/blog/nimbus-7b
...
AI-generated: selected, summarised and posted automatically by software, without human review.
------------------------------------------------------------------------
  ingest : fetched=12 in-batch duplicates=2 already stored=0 new=10
           source unreachable-feed failed (others continued): FileNotFoundError: feed file not found: does_not_exist.xml
  score  : scored=10 skipped=0
  verify : verified=8 skipped=0
  select : in window=8 blocked by verdict=3 not verified=0 briefed recently=0 cut by caps=1 selected=4
  brief  : 4 items in English, stored as brief #1

Run 3 on a fresh database with a model plan that has run out
  ingest : fetched=12 in-batch duplicates=2 already stored=0 new=10
  stopped: model plan exhausted (demo-exhausted) until 2026-03-11 10:45 UTC - usage limit reached, resets in 45 minutes

Run 4 ten minutes later: the guard skips the run instead of calling the model again
  stopped: model plan exhausted (demo-exhausted) until 2026-03-11 10:45 UTC - ... (run skipped)

demo OK
```

The three blocked items are the hype post, the leaked-rumour post and the partnership press release. The tracking-tagged re-post and the re-syndicated copy never reach the store. The link in the brief is the canonical one: `?utm_source=rss` is gone.

## Running on real feeds

```bash
cp config.example.yaml config.yaml          # adjust; every key is optional
ai-news-pipeline sources --config config.yaml
ai-news-pipeline run --config config.yaml   # exit 0 ok, 75 stopped by quota, 2 config error
```

`sources.yaml` ships five public feeds (Hugging Face, OpenAI, Google DeepMind, Simon Willison, Hacker News filtered to LLM stories) plus arXiv cs.CL, disabled because of its volume. Run it from cron or a systemd timer; the store makes repeated runs idempotent and the repost cooldown keeps an item out of two briefs.

### Choosing the model backend

| backend | how | notes |
|---|---|---|
| `mock` | default | keyword heuristics, deterministic, offline |
| `anthropic` | `pip install ".[anthropic]"`, `ANP_LLM_BACKEND=anthropic` | official SDK, structured outputs (`output_config.format` with a JSON schema); credentials from the SDK's usual environment variables; models and effort per stage in config |
| `command` | `ANP_LLM_BACKEND=command ANP_LLM_COMMAND="my-cli --json"` | any program that reads the prompt on stdin and prints JSON: a subscription CLI, a local model runner, a script |

Quota handling: a 402 billing error or a 429 that survives the SDK's own retries (anthropic), or a failed command that exits with 75 or ends its stderr with a "usage limit / rate limit" message, raises `QuotaExhausted`. The run stops, the stop is written to `state_dir/quota_stop.json` with the provider's reset time when given, and runs before that time exit at once with code 75. A stop never lasts longer than 24 hours, whatever the message says.

### Output language

`output_language: German` (or `ANP_OUTPUT_LANGUAGE=German`) goes into the compose prompt; translate the `categories` labels in config to match. Scoring and verification prompts stay in English on purpose: one set of prompts to evaluate.

### Telegram sink

Add `{type: telegram}` to `sinks` and set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the environment. The sink sends plain text (no parse mode, so no escaping bugs) with link previews off, splits at paragraph boundaries under the 4096-character limit, refuses a token that is not in the `digits:letters` shape and never puts the token in an error message.

### Environment variables

| variable | meaning |
|---|---|
| `ANP_CONFIG` | config file path (instead of `--config`) |
| `ANP_SOURCES`, `ANP_DB_PATH`, `ANP_STATE_DIR` | override paths |
| `ANP_OUTPUT_LANGUAGE` | brief language |
| `ANP_LLM_BACKEND`, `ANP_LLM_COMMAND` | model backend |
| `ANP_LLM_COMMAND_ENV` | comma list of env var names the `command` backend may see (its own login, a proxy) |
| `ANP_SINKS` | comma list, for example `console,file` |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | only for the telegram sink |

## Eval: scoring and verification behaviour

`src/ai_news_pipeline/data/eval_cases.yaml` holds 22 synthetic items labelled by hand with the verdict a careful editor would give and a relevance band. `ai-news-pipeline eval` runs them through the same prompts, schemas and input builders as the pipeline and prints:

```text
eval backend=mock cases=22 errors=0
  verdict accuracy   100%
  false-pass rate    0%  (non-pass items the gate lets through)
  relevance accuracy 86%  (band high >= 6)
  per verdict (precision / recall / support):
    dubious  1.00 / 1.00 / 3
    hype     1.00 / 1.00 / 3
    pass     1.00 / 1.00 / 8
    thin     1.00 / 1.00 / 8
```

Read the mock numbers as a check of the harness, not as a quality claim: the mock's keyword lists were written next to these cases. It still earned its place. The first run showed 96 % accuracy and one false pass: "LLM agents to clients" counted as an artefact because `cli` matched inside `clients`. Whole-word matching fixed it, and a test pins it. The numbers that matter come from a real backend:

```bash
ai-news-pipeline eval --backend anthropic --json-out eval.json --max-false-pass 0.1
```

The exit code is 1 when accuracy drops below `--min-accuracy` or false passes rise above `--max-false-pass`, so the eval can gate a prompt or model change in CI. With 22 cases one miss moves accuracy by about 4.5 points; treat differences smaller than that as noise.

## Design notes

- **Typed JSON, checked locally.** The Anthropic backend asks for structured output; the schema sent to the API keeps only the structure, and bounds such as 0..10 are checked by `validate()` for every backend. A model answering 11 is rejected the same way everywhere, and the item is skipped with a logged cause.
- **The model never writes links.** It returns ids; URLs and source names come from the store, and the store only accepts plain http(s) feed links made of RFC 3986 characters (no `javascript:`, `data:`, `user:password@`, quotes, angle or square brackets, or bidi characters; a feed link with raw non-ASCII characters is skipped too). Invented or duplicate ids are dropped and counted. A headline or blurb that contains a URL, anything that looks like a host name (`evil.com`, `evil.solutions`) or an `@handle` anyway is treated like a banned phrase (one retry, then the item is dropped), and every model field is collapsed to one line, so injected text cannot forge a `Source:` line.
- **Untrusted text stays data.** Article fields go in as JSON fields, and every system prompt says not to follow instructions found in them. The `command` backend runs with an allowlisted environment (PATH, HOME, locale, plus the names you list in `llm.command_env`), so the pipeline's own secrets never reach it. Quota is read only from a failed call's exit code 75 or its last stderr lines, so an article about rate limits cannot stop the pipeline.
- **Cheap model first, strict model on the top K.** Scoring runs on every item with a small model at low effort; verification runs only on the best candidates. The brief fails closed: an item verification never judged (below the top K, or its call failed) stays out unless `brief.require_verified: false`.
- **Feed dates are not trusted.** A publication date in the future is clamped to the ingest time, so a feed cannot keep an item inside every later brief window.
- **No silent no-ops.** Every skip has a reason in the run report; a run that publishes nothing says why (empty window, all blocked, cooldown, composition error).
- **Disclosure line.** Every brief ends with a fixed line saying it was generated and posted by software without human review (configurable).

## Project layout

```text
src/ai_news_pipeline/
  cli.py, pipeline.py, config.py, store.py, quota.py, evals.py, models.py
  sources/   base.py (canonical URL, hash), rss.py, registry.py
  stages/    dedupe.py, score.py, verify.py, brief.py, common.py
  llm/       base.py (contract, validator), mock.py, anthropic_backend.py, command_backend.py
  sinks/     console.py, file.py, telegram.py
  data/      demo feeds, demo sources, eval cases
tests/       one file per module, plus pipeline, CLI and SDK wire tests
sources.yaml, config.example.yaml, Dockerfile
```

## Honest limits

- The mock backend does not translate and its summaries are first sentences; it exists to run the system, not to write good briefs.
- The Anthropic backend is tested against the real SDK and a local fake endpoint, not against the live API (no key in CI). Run `eval --backend anthropic` once before trusting a model or prompt change.
- The link filter on model text fails closed: any dotted name ending in two or more letters (any script, punycode included) counts as a host unless the ending is a file extension that is not also a TLD, such as `.js`, `.cpp` or `.json` (so `README.md` and `main.py` count, because .md and .py are country domains), after NFKC normalisation, removal of zero-width characters and combining marks and mapping of lookalike dots; schemes, `www.`, bare IPv4 addresses, e-mail addresses, `@handles`, Markdown links and HTML tags are rejected too. The price is false hits on code names like `torch.compile` or `main.py`, which cost one retry of the brief call. Feed links themselves are checked separately at ingest.
- 22 eval cases are enough to catch regressions in gate behaviour, not to rank models.
- Only RSS/Atom adapters are included. The interface is one class with one method, so an API-based source (Hacker News API, GitHub trending) is a short addition.
- SQLite and one process: fine for hundreds of items per run, not built for many concurrent writers.
- The Dockerfile was not built on the machine this was written on (no Docker there); the Python path above is the tested one.

## License

MIT, see [LICENSE](LICENSE).
