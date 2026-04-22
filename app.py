"""
E-Commerce Product Search API.
Supports vector search, text search, hybrid search, and hybrid + rerank.
Includes mock cart & checkout for workshop interactivity.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
import voyageai

load_dotenv()

db_client: MongoClient = None
coll = None
orders_coll = None
vo: voyageai.Client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, coll, orders_coll, vo
    db_client = MongoClient(os.environ["MONGODB_URI"])
    db = db_client["workshop"]
    coll = db["products"]
    orders_coll = db["orders"]
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    yield
    db_client.close()


app = FastAPI(title="Product Search API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_query_embedding(text: str) -> list[float]:
    result = vo.embed([text], model="voyage-4-large", input_type="query")
    return result.embeddings[0]


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(vector|text|hybrid|rerank)$"),
    category: Optional[str] = None,
    limit: int = Query(12, ge=1, le=50),
):
    if mode == "vector":
        results = vector_search(q, category, limit)
    elif mode == "text":
        results = text_search(q, category, limit)
    elif mode == "rerank":
        results = hybrid_rerank_search(q, category, limit)
    else:
        results = hybrid_search(q, category, limit)
    return {"query": q, "mode": mode, "count": len(results), "results": results}


def vector_search(query: str, category: Optional[str], limit: int):
    query_vector = get_query_embedding(query)
    vs_stage = {
        "$vectorSearch": {
            "index": "genaiworkshop_index",
            "path": "description_embedding",
            "queryVector": query_vector,
            "numCandidates": max(100, limit * 10),
            "limit": limit,
        }
    }
    if category:
        vs_stage["$vectorSearch"]["filter"] = {"category": category}

    pipeline = [
        vs_stage,
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"description_embedding": 0}},
    ]
    return _serialize(coll.aggregate(pipeline))


def text_search(query: str, category: Optional[str], limit: int):
    compound: dict = {
        "must": [
            {"text": {"query": query, "path": ["name", "description"]}}
        ]
    }
    if category:
        compound["filter"] = [{"text": {"query": category, "path": "category"}}]

    pipeline = [
        {"$search": {"index": "text_search_index", "compound": compound}},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        {"$project": {"description_embedding": 0}},
        {"$limit": limit},
    ]
    return _serialize(coll.aggregate(pipeline))


def hybrid_search(query: str, category: Optional[str], limit: int, k: int = 60):
    vector_results = vector_search(query, category, limit=50)
    text_results = text_search(query, category, limit=50)

    rrf: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        did = str(doc.get("id", rank))
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)
        doc_map[did] = doc

    for rank, doc in enumerate(text_results):
        did = str(doc.get("id", rank))
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)
        doc_map[did] = doc

    sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:limit]
    results = []
    for did in sorted_ids:
        doc = doc_map[did]
        doc["score"] = round(rrf[did], 6)
        results.append(doc)
    return results


def hybrid_rerank_search(query: str, category: Optional[str], limit: int):
    candidates = hybrid_search(query, category, limit=30)
    if not candidates:
        return candidates
    descriptions = [doc.get("description", doc.get("name", "")) for doc in candidates]
    reranked = vo.rerank(query=query, documents=descriptions, model="rerank-2.5", top_k=limit)
    results = []
    for r in reranked.results:
        doc = candidates[r.index]
        doc["score"] = round(r.relevance_score, 4)
        results.append(doc)
    return results


def _serialize(cursor) -> list[dict]:
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        for k, v in doc.items():
            if isinstance(v, float):
                doc[k] = round(v, 4)
        results.append(doc)
    return results


class CartItem(BaseModel):
    name: str
    price: float
    category: str = ""
    brand: str = ""
    quantity: int = 1


class CheckoutRequest(BaseModel):
    user_name: str
    items: list[CartItem]
    search_mode: str = "hybrid"


@app.post("/api/checkout")
def checkout(req: CheckoutRequest):
    if not req.items:
        return {"error": "Cart is empty"}

    total = round(sum(item.price * item.quantity for item in req.items), 2)
    order_doc = {
        "user_name": req.user_name,
        "items": [item.model_dump() for item in req.items],
        "total": total,
        "item_count": sum(item.quantity for item in req.items),
        "search_mode_used": req.search_mode,
        "created_at": datetime.now(timezone.utc),
    }
    result = orders_coll.insert_one(order_doc)
    order_doc["_id"] = str(result.inserted_id)
    return {"order_id": order_doc["_id"], "total": total, "item_count": order_doc["item_count"]}


@app.get("/api/orders")
def get_orders(user_name: str = Query(...)):
    cursor = orders_coll.find({"user_name": user_name}).sort("created_at", -1).limit(20)
    orders = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc and isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        orders.append(doc)
    return {"orders": orders}


@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html") as f:
        return f.read()
