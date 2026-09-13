"""FastAPI application serving POST /ask endpoint using compiled LangGraph."""

import os
from fastapi import FastAPI
from schemas import QueryRequest, QueryResponse
from graph import app_graph

app = FastAPI(title="Zepto Customer Support AI Service")


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Zepto GenAI Support Assistant", "mock_mode": os.getenv("MOCK_LLM", "1") == "1"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": "",
        "context_chunks": [],
        "context_ids": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }

    final_state = app_graph.invoke(initial_state)

    return QueryResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)