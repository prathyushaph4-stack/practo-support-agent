Chunking strategy recommendation: Based on evaluation across the same 5 queries, fixed_size_collection achieved precision=1.000 and recall=1.000, while sentence_collection achieved precision=0.700 and recall=1.000. Both strategies achieved perfect recall (the correct document was always retrieved), but sentence-based chunking's shorter, more atomic chunks caused 3/5 queries to also surface an unrelated document in the top-3, lowering its precision to 0.700. We recommend deploying fixed_size_collection for this use case, since higher precision means fewer irrelevant chunks reach the generation step, reducing the risk of the agent blending in unrelated policy details.
Escalation Score Formula:
  escalation_score = 0.5 × follow_up_component + 0.5 × recency_component
  follow_up_component = 1.0 if follow_up_required else 0.0
  recency_component = days_since_created / 30

Threshold: escalation_score > 0.4567

Justification: 0.4567 is the 80th percentile of the actual escalation_score distribution across all 50 generated appointments (min=0.0333, max=0.9667, avg=0.3187). This flags the top ~20% most urgent cases for escalation, which our validation confirms: exactly 10/50 records (20.0%) are recommended for escalation, directly matching the intended percentile cut.
MOCK_LLM implementation notes:
- Extends crewai.llms.base_llm.BaseLLM directly (not litellm interception).
- Pitfall 1 (template "Observation:" trap): guarded by only counting
  Observation matches up to the number of Actions WE have emitted so far,
  and explicitly filtering out the literal template string
  "the result of the action".
- Pitfall 2 (tool name substring matching): guarded by reading each tool's
  args_schema.model_fields (or .args) to determine its real argument names,
  never inspecting tool.name. Additionally uses a two-pass scan across ALL
  available tools (record_id pattern first, query fallback second) so that
  tool list ORDER also cannot cause a wrong dispatch.
- Verified with 4 self-tests in crew/mock_llm.py's __main__ block, including
  asserts that fail loudly if either pitfall guard breaks.
  MockLLM real-world integration notes:
- CrewAI's default ReAct-style ".call()" interface does NOT pass tool
  schemas via the documented `tools=` parameter (confirmed empirically:
  it always arrived as None). Tool name/schema/description are instead
  embedded as plain text in the system prompt ("Tool Name: ...",
  "Tool Arguments: {...}", "Tool Description: ..."), discovered via
  file-based debug logging of real CrewAI calls.
- For tool-less agents (e.g. Response Composer), CrewAI instead embeds
  prior results directly into the user message as "This is the context
  you're working with: ...". MockLLM detects the tools=[] case and reads
  this block directly rather than attempting tool dispatch.
- The one remaining known imperfection: fixed-size chunking (Task 3) can
  produce chunks that begin mid-word (e.g. "uled time..." from "scheduled
  time"), since it cuts by raw character count rather than word boundary.
  This is a disclosed trade-off of the chosen chunking strategy, not a
  functional bug -- retrieval still correctly identifies the right
  source document in all demonstrated cases.
  Session Memory (Task 8):
- Implemented via LangChain's InMemoryChatMessageHistory +
  RunnableWithMessageHistory, keyed by session_id, in-process only
  (satisfies "does not need to survive a restart").
- Transcript A (session "patient-123"): Turn 2 correctly shows 2 prior
  messages (the Turn 1 human question + AI answer) automatically injected
  into context before the crew runs, proving multi-turn memory works.
- Transcript B (session "patient-456", freshly created while patient-123
  already had history): Turn 1 correctly shows 0 prior messages, proving
  sessions are isolated and memory does not leak across conversations.
- The LangChainDeprecationWarning pointing to LangGraph's persistence
  layer appeared as expected per the assignment brief and was left
  unsilenced.
  Structured Output Schema (Task 9):
- PracticeResponse (Pydantic BaseModel) with fields: query, answer, grounded,
  source_tool, escalation_recommended.
- Every crew response is passed through build_and_validate_response(),
  which classifies the raw output (RAG answer / status-lookup dict /
  fallback message) and constructs+validates a PracticeResponse instance.
- Verified: a policy question correctly produces grounded=true,
  source_tool="rag_lookup"; an appointment lookup correctly produces
  source_tool="check_appointment_status" with escalation_recommended
  correctly extracted from the tool's real output; a deliberately invalid
  type (grounded="not_a_bool") correctly raises pydantic.ValidationError.
  Guardrails (Task 10):
1. PII masking: contact numbers (fixed 10-digit Indian mobile format,
   optionally +91-prefixed) are regex-masked to [CONTACT_NUMBER_REDACTED]
   before any agent sees the input. Patient name, diagnosis, and insurance
   ID are explicitly out of scope for masking (free text, no universal
   format), per the assignment's own scoping.
2. Prompt-injection detection: input is checked against known injection
   phrasings (e.g. "ignore all previous instructions", "reveal your
   system prompt"). A detected injection blocks the request entirely
   BEFORE any CrewAI agent executes -- confirmed via real pipeline run
   showing zero agent/tool execution for a blocked request.
3. Output-side groundedness check: reuses the Task 4 empirically-
   calibrated similarity threshold (0.3162) to refuse answering when
   the top retrieved chunk's similarity falls below it.
Demonstrated firing in the real crew pipeline (not just in isolation):
a contact number was masked before reaching the Retrieval Agent (visible
in the agent's own Task text), and a prompt-injection attempt was blocked
with zero agent execution.
FastAPI Deployment (Task 11):
- POST /ask: accepts a question, routes to RAG or appointment-lookup tool
  automatically based on presence of a record ID pattern, returns a
  validated PracticeResponse.
- POST /add-document: adds a new KB document at runtime and immediately
  re-indexes it into fixed_size_collection; demonstrated live by adding a
  parking-policy document and confirming a subsequent /ask query
  retrieved it correctly.
- WS /ws/chat: real-time multi-turn chat over one persistent connection.
  Known issue found and fixed during testing: CrewAI's crew.kickoff() is
  a synchronous, blocking call and cannot be invoked directly inside an
  async WebSocket handler (CrewAI raises a RuntimeError to prevent
  blocking the event loop). Fixed by running run_full_pipeline() inside
  Starlette's run_in_threadpool(), keeping the event loop responsive.
- Client disconnects are caught via WebSocketDisconnect and logged; the
  server continues running and was demonstrated serving a second,
  independent client session immediately afterward with a fresh
  session_id, proving no crash or restart was needed.
  Structured Logging (Task 12):
- Every /ask, /add-document, and /ws/chat request writes one JSON-Lines
  entry to logs/requests.jsonl containing: trace_id, timestamp, endpoint,
  request_text, pii_masked flag, response_summary, and duration_ms.
- Reuses Task 10's mask_pii() directly (not reimplemented) to mask
  request_text before writing to disk, so a fixed-format PII field never
  reaches the log file in the clear.
- Verified in a live request: a raw contact number in the request body
  was correctly masked to [CONTACT_NUMBER_REDACTED] in the actual log
  file on disk, confirmed via direct file read.
- Known limitation: documents added via POST /add-document exist only in
  that server process's in-memory ChromaDB collection and do not persist
  across a server restart, since kb/documents.py's DOCUMENTS list is
  appended to at runtime only, not written back to disk. This is a
  reasonable scope boundary for a MOCK_LLM, zero-network capstone.
  LLM-as-Judge Evaluation (Task 13):
- 15-query test set: 1 query per required KB topic (12), 2 deliberately
  out-of-scope queries, 1 appointment-lookup edge case.
- Since MOCK_LLM has no real model to reason with, the "judge" is a
  deterministic rule-based scorer: Accuracy = keyword overlap with the
  expected source document; Grounding = retrieval similarity relative to
  the Task 4 threshold (or correct refusal for out-of-scope); Completeness
  = answer substance vs. bare fallback; Safety = absence of leaked PII or
  echoed injection patterns.
- Averages across 15 queries: Accuracy=0.696, Grounding=1.000,
  Completeness=1.000, Safety=1.000.
- Notable finding: "How do I book an appointment?" scored Accuracy=0.10.
  Investigation showed the query retrieved a chunk from KB008 (emergency-
  visit protocol) instead of KB001 (appointment-booking), because KB008's
  chunk happens to contain overlapping vocabulary ("appointment",
  "booking") near a chunk boundary. This is an honest limitation of
  fixed-size chunking + cosine similarity on short policy texts with
  shared vocabulary across documents, consistent with the retrieval
  imperfections already observed in Task 4/5. It is disclosed here rather
  than hidden or "fixed" by altering the test query, since surfacing real
  retrieval weaknesses is the intended purpose of this evaluation.
  Autogen Review Stage (Task 14):
- 2-agent RoundRobinGroupChat (Policy_Compliance_Reviewer, Final_Editor),
  bounded with MaxMessageTermination(3) (correctly accounts for the
  initiating task message as message 1, allowing both agents to speak).
- Final_Editor uses output_content_type=ReviewVerdict; the Team is
  constructed with custom_message_types=[StructuredMessage[ReviewVerdict]]
  to avoid the "Message type is not registered" crash.
- MockAutogenClient implements autogen_core.models.ChatCompletionClient
  as a deterministic rule-based judge (no API keys/network), mirroring
  MockLLM's role for CrewAI.
- Demonstrated on 2 sample queries: a fully grounded draft was approved
  unchanged; a draft with a deliberately injected ungrounded absolute
  claim ("...guaranteed to work for everyone") was correctly flagged and
  revised, with the offending sentence removed cleanly.
- Installed autogen-agentchat/autogen-core/autogen-ext at version 0.7.5
  (an earlier pip resolution had picked up abandoned 0.0.x placeholder
  packages under the same names; fixed by installing with an explicit
  >=0.4 version floor).
  AI Governance (Task 15):
1. Application layer (least autonomy): verified via direct inspection of
   the real agent objects in crew/agents.py -- retrieval_agent has only
   [rag_lookup], lookup_agent has only [check_appointment_status],
   composer_agent has no tools at all. check_appointment_status is never
   wired to any agent except lookup_agent, so no other agent has a code
   path to invoke it, regardless of prompt content.
2. Risk classification: High. Justification: the system handles
   healthcare-adjacent appointment records and produces escalation
   recommendations affecting patient follow-up care, placing it in the
   assignment's "High: medical data" tier rather than Medium or Low.
3. Runtime layer: a per-request token budget cap (200 tokens, estimated
   via a ~4-chars-per-token heuristic) is enforced. A normal 8-token
   request is accepted; a deliberately oversized 298-token request is
   correctly rejected with BudgetExceededError rather than silently
   processed.

Known environment note: installing autogen-ext downgraded protobuf to
5.29.6, which broke chromadb/crewai's import chain (needs >=6.33.5).
Fixed by upgrading protobuf to 7.36.1. pip reports declared version-range
conflicts for autogen-core and opentelemetry-proto against this protobuf
version, but both libraries were verified to import and run correctly
end-to-end at runtime.
Response Caching (Task 16):
- In-memory cache keyed by normalized query text (lowercased, whitespace-
  collapsed) wraps grounded_answer().
- Demonstrated: Call 1 (cache miss) took 467.43ms and incremented a real-
  call counter to 1. Call 2 (identical query) was a cache hit taking
  0.10ms, with the real-call counter remaining at 1 -- proving the
  underlying embedding + ChromaDB search was genuinely skipped, not just
  fast. Call 3 (same query, different casing/whitespace) also correctly
  hit the cache, proving normalization works. Overall speedup: ~4853x on
  a cache hit.