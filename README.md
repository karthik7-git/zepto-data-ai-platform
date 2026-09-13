# Zepto Data & AI Platform

An end-to-end production-grade AI/ML system built for Zepto's analytics guild. The platform integrates catalog data engineering, customer behavior predictive modeling, and a grounded GenAI customer support service into a unified repository.

---

## 1. Project Architecture & Setup

### Repository Layout
```text
zepto-data-ai-platform/
├── README.md                          # Master documentation & setup guide
├── requirements.txt                   # Consolidated project dependencies
├── data_pipeline/
│   ├── README.md                      # Pipeline setup & cleaning specifications
│   ├── scraper.py                     # BeautifulSoup scraper (>60 items, 4 categories)
│   ├── database.py                    # Normalized SQLite schema & loader
│   ├── query_analysis.py              # 5 SQL queries + Pandas equivalence test
│   └── zepto_catalog.db               # SQLite database file
├── analytics/
│   ├── README.md                      # Full EDA story & model comparison report
│   ├── titanic.csv                    # Committed offline fallback dataset (single load)
│   ├── 01_eda.py                      # Profiling, missing handling, skewness, 4 plots
│   ├── 02_modeling.py                 # Leak-free pipeline, classifiers, SMOTE, regression
│   └── best_pipeline.joblib           # Serialized end-to-end pipeline artifact
└── support_assistant/
    ├── README.md                      # RAG architecture & verification transcripts
    ├── Dockerfile                     # Container deployment specification
    ├── docs/                          # 8 verified Zepto policy text documents
    │   ├── doc_01.txt
    │   └── ... doc_08.txt
    ├── ingest.py                      # SentenceTransformers + ChromaDB ingestion
    ├── schemas.py                     # Pydantic input/output validation models
    ├── prompt.py                      # Role-Context-Task-Format-Length structured prompt
    ├── graph.py                       # LangGraph StateGraph (3 nodes + conditional edge)
    └── main.py                        # FastAPI application serving POST /ask