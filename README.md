# Practo Support Agent — Final Capstone

**Track completed: Practo (Healthcare)**

This project is a multi-component AI agent system for Practo's patient-experience
team, built entirely under `MOCK_LLM` mode — zero API keys, zero network calls for
any language-model reasoning. It combines a RAG knowledge base, a CrewAI multi-agent
crew, an Autogen review stage, guardrails, evaluation, governance, and a FastAPI
deployment into one working system.

## Running This Project

- All language-model reasoning (grounded generation judge, CrewAI crew, Autogen
  review team) runs under a deterministic, rule-based `MOCK_LLM` — no API keys or
  internet access required for any reasoning step. (SentenceTransformers does
  download its embedding model from Hugging Face on first run only, which requires
  one-time internet access; ChromaDB and all reasoning are fully local/offline.)
- **Before running any CrewAI crew code**, the following environment variables are
  set (see `.env` / `crew/agents.py`), to prevent CrewAI's telemetry from making an
  outbound network call:
  ```
  CREWAI_DISABLE_TELEMETRY=true
  OTEL_SDK_DISABLED=true
  ```
- Virtual environment: Python 3.13, `venv` (see `requirements.txt` for all
  dependencies: crewai, chromadb, sentence-transformers, langchain, fastapi,
  uvicorn, pydantic, autogen-agentchat, autogen-core, autogen-ext, websockets).

---

## Part 1 — Dataset Design & RAG Core

### Task 1 — Appointment Dataset (`data/dataset.py`)

**Design choices (for deterministic reproduction):**
- `SEED = 42`
- `NUM_RECORDS = 50`
- Category weights: equal (20% each) across General Medicine, Cardiology,
  Dermatology, Pediatrics, Orthopedics
- Status weights: equal (20% each) across Scheduled, Completed, Cancelled,
  No-Show, Rescheduled
- `FOLLOW_UP_PROBABILITY = 0.20` (sits mid-band of the required 10–30% range)
- Consultation fee ranges (INR), reasoning: fees scale with specialty
  equipment/expertise needs:
  - General Medicine: 300–600
  - Pediatrics: 350–650
  - Dermatology: 500–900
  - Orthopedics: 800–1500
  - Cardiology: 900–1800

**Verified result:** 50 records generated; every category ≥3 records; every status
≥1 record; `follow_up_required=True` percentage landed within the required 10–30%
band. All structural assertions pass on every run with `SEED=42`.

### Task 2 — Knowledge Base (`kb/documents.py`)

12 documents authored, 2–5 sentences each, covering all required topics:
appointment-booking, cancellation/rescheduling, consultation-fee structure,
insurance-claim process, prescription-refill policy, lab-test turnaround,
telemedicine eligibility, emergency-visit protocol, patient-data privacy,
follow-up-visit discount, second-opinion process, home-visit eligibility.

### Task 3 — Chunking & Indexing (`rag/chunking.py`, `rag/indexer.py`)

- **Fixed-size chunking**: 120-character windows, 30-character overlap → **60 chunks**
- **Sentence-based chunking**: split on sentence boundaries (regex handles missing
  spaces after periods: `(?<=[.!?])\s*(?=[A-Z])`) → **36 chunks**
- Both embedded with `all-MiniLM-L6-v2` (SentenceTransformers, free/local) and
  indexed into **two separate ChromaDB collections**, both explicitly configured
  with `metadata={"hnsw:space": "cosine"}` so similarity scores are true cosine
  similarity (0–1 range), not ChromaDB's default L2 distance.

### Task 4 — Grounded Generation + Threshold Calibration (`rag/retriever.py`)

Empirical calibration measured top-1 cosine similarity for 4 in-scope and 3
out-of-scope queries on `sentence_collection`:

| Type | Query | Similarity |
|---|---|---|
| In-scope | cancellation policy | 0.5224 |
| In-scope | cardiology cost | 0.6046 |
| In-scope | prescription refill | 0.8084 |
| In-scope | insurance claim | 0.5854 |
| Out-of-scope | capital of France | 0.0290 |
| Out-of-scope | pizza recipe | 0.0457 |
| Out-of-scope | weather in Bangalore | 0.1100 |

Clusters were cleanly separated (lowest in-scope 0.5224 vs. highest out-of-scope
0.1100 — no overlap). **Chosen threshold: 0.3162** (midpoint). This is deliberately
NOT a tutorial-default preset (0.5/0.6/0.7) — a 0.5 threshold would have wrongly
rejected our weakest in-scope query (0.5224).

Demonstrated on 5 in-scope queries (all correctly grounded) + 1 deliberately
out-of-scope query ("What is the capital of France?" — correctly triggered the
"I don't know" fallback).

### Task 5 — Precision/Recall Evaluation & Recommendation

Computed on the same 5 queries, document-level, deduplicated, top-k=3:

| Collection | Precision | Recall |
|---|---|---|
| `fixed_size_collection` | 1.000 | 1.000 |
| `sentence_collection` | 0.700 | 1.000 |

**Recommendation:** Both strategies achieved perfect recall, but
`sentence_collection`'s shorter, more atomic chunks caused 3/5 queries to also
surface an unrelated document in the top-3, lowering its precision to 0.700. We
recommend deploying **`fixed_size_collection`** for this use case, since higher
precision reduces the risk of blending in unrelated policy details. **This
collection is used as the fixed, permanent input for all of Part 2 onward**, per
the assignment's instruction to lock in the Task 5 recommendation.

---

## Part 2 — CrewAI Orchestration, Memory & Guardrails

### Task 6 — Escalation Score Tool (`tools/lookup_tool.py`)

**Formula:**
```
escalation_score = 0.5 × follow_up_component + 0.5 × recency_component
follow_up_component = 1.0 if follow_up_required else 0.0
recency_component   = days_since_created / 30
```

**Threshold:** `escalation_score > 0.4567` — set at the **80th percentile of the
actual escalation_score distribution** across all 50 generated appointments
(min=0.0333, max=0.9667, avg=0.3187). This flags the top ~20% most urgent cases;
validated result: exactly **10/50 records (20.0%)** recommended for escalation,
directly matching the intended percentile cut.

### Task 7 — CrewAI 3-Agent Crew (`crew/agents.py`, `crew/mock_llm.py`)

Three agents: **Retrieval Agent** (tool: `rag_lookup`), **Lookup Agent** (tool:
`check_appointment_status`), **Response Composer** (no tools — combines prior
agents' outputs). Both tools were demonstrably invoked via real `.kickoff()` calls
on different queries, with tool-execution traces as evidence.

**`MOCK_LLM` implementation notes** (extends `crewai.llms.base_llm.BaseLLM`):
- **Pitfall 1 guard** (template "Observation:" trap): the literal line
  `"Observation: the result of the action"` lives inside CrewAI's own system-prompt
  template as a format example. Guarded by only counting `Observation:` matches up
  to the number of `Action:` lines WE have generated so far in the real conversation
  (never scanning the system-prompt template itself), plus explicitly filtering out
  the literal template string.
- **Pitfall 2 guard** (tool-name substring matching): tool dispatch never inspects
  a tool's name string. **Key discovery via debug logging:** CrewAI's default
  ReAct-style agents do NOT pass tool schemas via the documented `tools=` parameter
  (confirmed empirically — it always arrived as `None`). Tool name/schema/description
  are instead embedded as plain text in the system prompt
  (`Tool Name: ...` / `Tool Arguments: {...JSON schema...}` / `Tool Description: ...`).
  `MOCK_LLM` parses this directly from the system-prompt text via regex, and
  dispatches based on the tool's declared JSON-schema argument names (`record_id`
  vs. `query`), never the tool's name. A two-pass scan (record_id pattern first,
  query fallback second) across ALL available tools also prevents tool list ORDER
  from causing a wrong dispatch.
- **Composer/no-tools case:** For agents with zero tools (e.g. Response Composer),
  CrewAI instead embeds prior results into the user message as
  `"This is the context you're working with: ..."`. `MOCK_LLM` detects the
  zero-tools case and extracts this block directly as the final answer, rather than
  attempting (and failing) tool dispatch.
- The one remaining known imperfection: fixed-size chunking (Task 3) can produce
  chunks that begin mid-word (e.g. `"uled time..."` from `"...scheduled time"`),
  since it cuts by raw character count rather than word boundary. Disclosed
  trade-off, not a functional bug — retrieval still correctly identifies the right
  source document in all demonstrated cases.

### Task 8 — Session Memory (`crew/memory.py`)

LangChain `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`, keyed by
`session_id`, in-process only (satisfies "does not need to survive a restart").

- **Transcript A** (session `patient-123`): Turn 2 correctly shows 2 prior messages
  (Turn 1's human question + AI answer) automatically injected before the crew ran —
  proves multi-turn memory works.
- **Transcript B** (session `patient-456`, created while `patient-123` already had
  history): Turn 1 correctly shows 0 prior messages — proves sessions are isolated
  and memory does not leak across conversations.
- Final check: `patient-123` held 4 total messages, `patient-456` held 2 — exact
  match to expectations.
- The expected `LangChainDeprecationWarning` (pointing to LangGraph's persistence
  layer) appeared and was left unsilenced, per the assignment brief.

### Task 9 — Structured Output Schema (`crew/schema.py`)

Pydantic `PracticeResponse` model: `query`, `answer`, `grounded`, `source_tool`,
`escalation_recommended`. Every crew response is passed through
`build_and_validate_response()`, which classifies the raw output (RAG answer /
status-lookup dict / fallback) and constructs + validates a `PracticeResponse`
instance. Verified: a policy question → `grounded=true, source_tool="rag_lookup"`;
an appointment lookup → `source_tool="check_appointment_status"` with
`escalation_recommended` correctly extracted; a deliberately invalid type
(`grounded="not_a_bool"`) correctly raises `pydantic.ValidationError`.

### Task 10 — Guardrails (`guardrails/guardrails.py`)

1. **PII masking:** contact numbers (10-digit Indian mobile format, optionally
   `+91`-prefixed) are regex-masked to `[CONTACT_NUMBER_REDACTED]` before any agent
   sees the input. Patient name, diagnosis, and insurance ID are explicitly out of
   scope for masking (free text, no universal format), per the assignment's own
   scoping.
2. **Prompt-injection detection:** input checked against known injection phrasings
   (e.g. "ignore all previous instructions", "reveal your system prompt"). A
   detected injection blocks the request entirely **before any CrewAI agent
   executes** — confirmed via a real pipeline run showing zero agent/tool execution
   for a blocked request.
3. **Output-side groundedness check:** reuses the Task 4 empirically-calibrated
   threshold (0.3162) to refuse answering when the top retrieved chunk's similarity
   falls below it.

Demonstrated firing in the **real** crew pipeline, not just in isolation: a contact
number was masked before reaching the Retrieval Agent (visible in the agent's own
`Task:` text), and a prompt-injection attempt was blocked with zero agent execution.

---

## Part 3 — Evaluation, Observability & FastAPI Deployment

### Task 11 — FastAPI Deployment (`api/main.py`)

- **`POST /ask`** — routes to RAG or appointment-lookup tool automatically based on
  presence of a record-ID pattern; returns a validated `PracticeResponse`.
- **`POST /add-document`** — adds a new KB document at runtime and immediately
  re-indexes it into `fixed_size_collection`; demonstrated live by adding a
  parking-policy document and confirming a subsequent `/ask` query retrieved it
  correctly.
- **`WS /ws/chat`** — real-time multi-turn chat over one persistent connection.
- **Known issue found and fixed during testing:** CrewAI's `crew.kickoff()` is a
  synchronous, blocking call and cannot be invoked directly inside an async
  WebSocket handler (CrewAI raises a `RuntimeError` to prevent blocking the event
  loop). Fixed by running `run_full_pipeline()` inside Starlette's
  `run_in_threadpool()`.
- Client disconnects are caught via `WebSocketDisconnect` and logged; the server
  was demonstrated continuing to run and serving a second, independent client
  session immediately afterward with a fresh `session_id` — no crash or restart
  needed.
- **Known limitation:** documents added via `/add-document` exist only in that
  server process's in-memory ChromaDB collection and do not persist across a
  server restart, since `kb/documents.py`'s `DOCUMENTS` list is appended to at
  runtime only, not written back to disk. A reasonable scope boundary for a
  `MOCK_LLM`, zero-network capstone.

### Task 12 — Structured JSON-Lines Logging (`api/logging_utils.py`)

Every `/ask`, `/add-document`, and `/ws/chat` request writes one JSON-Lines entry
to `logs/requests.jsonl` containing: `trace_id`, `timestamp`, `endpoint`,
`request_text`, `pii_masked` flag, `response_summary`, `duration_ms`. **Reuses
Task 10's `mask_pii()` directly** (not reimplemented) so a fixed-format PII field
never reaches disk in the clear. Verified in a live request: a raw contact number
was correctly masked to `[CONTACT_NUMBER_REDACTED]` in the actual log file on disk
(confirmed via direct file read).

### Task 13 — LLM-as-Judge Evaluation (`eval/llm_judge_eval.py`)

15-query test set: 1 query per required KB topic (12) + 2 deliberately
out-of-scope queries + 1 appointment-lookup edge case. Since `MOCK_LLM` has no real
model to reason with, the "judge" is a deterministic rule-based scorer:
- **Accuracy** = keyword overlap between the answer and the expected source
  document
- **Grounding** = retrieval similarity relative to the Task 4 threshold (or correct
  refusal for out-of-scope queries)
- **Completeness** = answer substance vs. a bare fallback phrase
- **Safety** = absence of leaked PII or echoed injection patterns

**Averages across 15 queries:** Accuracy = 0.696, Grounding = 1.000,
Completeness = 1.000, Safety = 1.000.

**Notable finding (disclosed, not hidden):** "How do I book an appointment?"
scored Accuracy = 0.10. Investigation showed the query retrieved a chunk from
KB008 (emergency-visit protocol) instead of KB001 (appointment-booking), because
KB008's chunk happens to contain overlapping vocabulary ("appointment", "booking")
near a chunk boundary. This is an honest limitation of fixed-size chunking +
cosine similarity on short policy texts with shared vocabulary across documents,
consistent with retrieval imperfections already observed in Tasks 4/5.

---

## Part 4 — Resilience & Governance

### Task 14 — Autogen Review Stage (`review/autogen_review.py`)

2-agent `RoundRobinGroupChat` (`Policy_Compliance_Reviewer`, `Final_Editor`),
bounded with `MaxMessageTermination(3)` (correctly accounts for the initiating task
message as message 1, allowing both agents to speak). `Final_Editor` uses
`output_content_type=ReviewVerdict`; the Team is constructed with
`custom_message_types=[StructuredMessage[ReviewVerdict]]` to avoid the "Message
type is not registered" crash. `MockAutogenClient` implements
`autogen_core.models.ChatCompletionClient` as a deterministic rule-based judge (no
API keys/network), mirroring `MockLLM`'s role for CrewAI.

Demonstrated on 2 sample queries: a fully grounded draft was approved unchanged; a
draft with a deliberately injected ungrounded absolute claim
("...guaranteed to work for everyone") was correctly flagged and revised, with the
offending sentence cleanly removed.

**Environment note:** installed `autogen-agentchat`/`autogen-core`/`autogen-ext` at
version **0.7.5** (an earlier unpinned `pip install` had picked up abandoned `0.0.x`
placeholder packages under the same names on PyPI; fixed with an explicit `>=0.4`
version floor). Installing `autogen-ext` downgraded `protobuf` to 5.29.6, which
broke `chromadb`/CrewAI's import chain (needs `>=6.33.5`); fixed by upgrading
`protobuf` to 7.36.1. `pip` reports declared version-range conflicts for
`autogen-core` and `opentelemetry-proto` against this protobuf version, but both
libraries were verified to import and run correctly end-to-end at runtime.

### Task 15 — Four-Layer AI Governance (`governance/governance.py`)

1. **Application layer (least autonomy):** verified via direct inspection of the
   real agent objects — `retrieval_agent` has only `['rag_lookup']`, `lookup_agent`
   has only `['check_appointment_status']`, `composer_agent` has no tools at all.
   `check_appointment_status` is never wired to any agent except `lookup_agent`, so
   no other agent has a code path to invoke it, regardless of prompt content.
2. **Risk classification: High.** Justification: the system handles
   healthcare-adjacent appointment records (categories, status, fees) and produces
   escalation recommendations affecting patient follow-up care — placing it in the
   assignment's "High: medical data" tier rather than Medium or Low, since an
   incorrect escalation decision or leaked appointment record carries real-world
   harm potential.
3. **Runtime layer:** a per-request token budget cap (200 tokens, estimated via a
   ~4-characters-per-token heuristic) is enforced. A normal 8-token request is
   accepted; a deliberately oversized 298-token request is correctly rejected with
   `BudgetExceededError` rather than silently processed.

### Task 16 — Response Caching (`rag/cache.py`)

In-memory cache keyed by normalized query text (lowercased, whitespace-collapsed)
wraps `grounded_answer()`. **Demonstrated:** Call 1 (cache miss) took 467.43ms and
incremented a real-call counter to 1. Call 2 (identical query) was a cache hit
taking 0.10ms, with the real-call counter remaining at 1 — proving the underlying
embedding + ChromaDB search was genuinely skipped, not just fast. Call 3 (same
query, different casing/whitespace) also correctly hit the cache, proving
normalization works. **Overall speedup: ~4853x on a cache hit.**

---

## Repository Structure

```
practo-support-agent/
├── data/dataset.py              # Task 1
├── kb/documents.py               # Task 2
├── rag/
│   ├── chunking.py                # Task 3
│   ├── indexer.py                 # Task 3
│   ├── retriever.py                # Task 4
│   └── cache.py                   # Task 16
├── tools/
│   ├── lookup_tool.py             # Task 6
│   ├── rag_tool.py                 # Task 7 (CrewAI tool wrapper)
│   └── status_tool.py              # Task 7 (CrewAI tool wrapper)
├── crew/
│   ├── mock_llm.py                 # Task 7
│   ├── agents.py                   # Task 7, 9, 10
│   ├── memory.py                   # Task 8
│   └── schema.py                   # Task 9
├── guardrails/guardrails.py       # Task 10
├── api/
│   ├── main.py                     # Task 11
│   └── logging_utils.py            # Task 12
├── eval/
│   ├── evaluate.py                  # Task 5
│   └── llm_judge_eval.py            # Task 13
├── review/
│   ├── mock_autogen_client.py       # Task 14
│   └── autogen_review.py            # Task 14
├── governance/governance.py       # Task 15
├── logs/requests.jsonl             # Task 12 output
├── .env                             # MOCK_LLM / telemetry flags
├── requirements.txt
└── README.md                        # this file
```
