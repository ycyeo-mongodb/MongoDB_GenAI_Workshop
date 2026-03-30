# MongoDB GenAI Workshop

Build an AI-powered e-commerce product search application using **MongoDB Atlas Vector Search**, **Voyage AI embeddings**, and **hybrid search**.

## What You'll Build

A product search engine for a 1,000-item e-commerce catalog that supports:

- **Semantic Search** — Find products by meaning using `$vectorSearch`
- **Full-Text Search** — Classic keyword search using `$search`
- **Hybrid Search** — Combine both with Reciprocal Rank Fusion (RRF)
- **Reranking (Bonus)** — Refine results with Voyage AI's `rerank-2.5`

## Project Structure

```
├── data/
│   └── products.json           # 1,000-product sample catalog (8 categories)
├── scripts/
│   └── generate_catalog.py     # Script to regenerate the dataset
├── static/
│   └── index.html              # Search frontend (HTML/CSS/JS)
├── 01_load_and_embed.py        # Load products + generate embeddings
├── 02_create_indexes.py        # Create vector + text search indexes
├── 03_semantic_search.py       # Semantic search with $vectorSearch
├── 04_hybrid_search.py         # Hybrid search with RRF
├── 05_reranking.py             # (Bonus) Rerank with Voyage AI rerank-2.5
├── app.py                      # FastAPI search API (all 4 modes)
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

## Prerequisites

- Python 3.10+
- A [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) account (free tier works)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ycyeo-mongodb/MongoDB_GenAI_Workshop.git
cd MongoDB_GenAI_Workshop
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env with your MongoDB URI and Voyage AI API key
```

> **Voyage AI API key**: Create this in Atlas under **Services → AI Models → Create model API key**. No separate Voyage AI account needed.

### 3. Run the workshop steps

```bash
python 01_load_and_embed.py     # Load data + generate embeddings
python 02_create_indexes.py     # Create search indexes (wait until READY)
python 03_semantic_search.py    # Try semantic search queries
python 04_hybrid_search.py      # Try hybrid search queries
python 05_reranking.py          # (Bonus) Try reranking
```

### 4. Run the search app

```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) and try searching!

## Tech Stack

| Component | Technology |
|-----------|------------|
| Database | MongoDB Atlas (M0 Free Tier) |
| Embeddings | Voyage AI `voyage-4-large` (via Atlas) |
| Reranking | Voyage AI `rerank-2.5` (via Atlas) |
| Backend | Python / FastAPI |
| Frontend | HTML / CSS / JavaScript |

## Resources

- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
- [MongoDB Atlas Search](https://www.mongodb.com/docs/atlas/atlas-search/)
- [Voyage AI by MongoDB](https://www.mongodb.com/docs/voyageai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
