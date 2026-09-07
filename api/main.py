"""
main.py
Practo Capstone - Task 11: FastAPI Deployment
-------------------------------------------------
Exposes the crew via 2 HTTP endpoints + 1 WebSocket endpoint.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "crew"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "kb"))

import re
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from agents import run_full_pipeline
from schema import PracticeResponse
from documents import DOCUMENTS

# Reuse the SAME model + collection objects that rag_tool.py already built
# at import time (via agents.py). Do NOT call build_index() again here --
# it deletes and recreates the collections, which would invalidate the
# references rag_tool.py is already holding, causing a "Collection does
# not exist" error.
import rag_tool
import time

sys.path.append(os.path.dirname(__file__))
from logging_utils import log_request

app = FastAPI(title="Practo Support Agent API")

RECORD_ID_PATTERN = re.compile(r"\bAPT\d{4}\b", re.IGNORECASE)

_model = rag_tool._model
_fixed_collection = rag_tool._fixed_collection


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., description="The patient's question in plain English")


class AskResponse(BaseModel):
    result: PracticeResponse


class AddDocumentRequest(BaseModel):
    topic: str = Field(..., description="Short topic label for the new document")
    text: str = Field(..., description="The full policy document text (2-5 sentences)")


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_added: int
    message: str


# ---------------------------------------------------------------------------
# Endpoint 1: POST /ask
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start_time = time.time()
    needs_lookup = bool(RECORD_ID_PATTERN.search(request.question))
    result = run_full_pipeline(request.question, needs_lookup=needs_lookup)

    log_request(
        endpoint="/ask",
        request_text=request.question,
        response_summary=result.answer,
        start_time=start_time,
    )

    return AskResponse(result=result)


# ---------------------------------------------------------------------------
# Endpoint 2: POST /add-document
# ---------------------------------------------------------------------------

@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest):
    start_time = time.time()
    new_doc_id = f"KB{len(DOCUMENTS) + 1:03d}"
    new_doc = {"doc_id": new_doc_id, "topic": request.topic, "text": request.text}
    DOCUMENTS.append(new_doc)

    # Re-embed and add just this one new doc's fixed-size chunks to the
    # existing collection (avoids a full re-index for a single addition).
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
    from chunking import fixed_size_chunks

    new_chunks = fixed_size_chunks([new_doc])
    embeddings = _model.encode([c["text"] for c in new_chunks]).tolist()
    _fixed_collection.upsert(
        ids=[c["chunk_id"] for c in new_chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in new_chunks],
        metadatas=[{"doc_id": c["doc_id"]} for c in new_chunks],
    )

    log_request(
        endpoint="/add-document",
        request_text=f"{request.topic}: {request.text}",
        response_summary=f"Added {new_doc_id} with {len(new_chunks)} chunks",
        start_time=start_time,
    )
    
    return AddDocumentResponse(
        doc_id=new_doc_id,
        chunks_added=len(new_chunks),
        message=f"Document '{new_doc_id}' added and indexed successfully.",
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint: real-time multi-turn chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    await websocket.send_json({"event": "connected", "session_id": session_id})

    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")

            start_time = time.time()
            needs_lookup = bool(RECORD_ID_PATTERN.search(question))
            result = await run_in_threadpool(run_full_pipeline, question, needs_lookup)

            log_request(
                endpoint="/ws/chat",
                request_text=question,
                response_summary=result.answer,
                start_time=start_time,
            )

            await websocket.send_json({
                "event": "answer",
                "result": result.model_dump(),
            })
    
    except WebSocketDisconnect:
        # Client disconnected mid-conversation -- log it and keep the server
        # running for other clients. This is the required graceful handling.
        print(f"[WebSocket] Client {session_id} disconnected. Server continues running.")


@app.get("/")
def root():
    return {
        "message": "Practo Support Agent API is running.",
        "endpoints": ["POST /ask", "POST /add-document", "WS /ws/chat"],
    }