from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis
import json
from typing import List
from loguru import logger
import time
from src.config import (REDIS_HOST, REDIS_PORT, CACHE_CONFIG, REC_CONFIG, API_CONFIG)

redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        db=0, 
        decode_responses=True
    )
    logger.info("Connected to Redis")
    yield
    redis.client.close()
    logger.info("Disconnected from Redis")

app = FastAPI(
    title="Smart Recommendation Engine",
    description="Hybrid Recommendation System using LightFM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_popular_items() ->list[str]:
    cached = redis_client.get("popular_items")
    if cached:
        return json.load(cached)

    return [f"item_{i}" for i in range(REC_CONFIG["fallback_items"] + 1)]

@app.get("/health")
async def health_check():
    try:
        redis_ping = redis_client.ping()
        return {
            "status": "healty",
            "redis": "connected" if redis_ping else "disconnected",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/recommendations/{user_id}", response_model=List[str])
async def get_recommendations(user_id: str):
    start_time = time.time()
    cache_key = f"{CACHE_CONFIG['prefix']}: {user_id}"

    cached_recs = redis_client.get(cache_key)

    if cached_recs:
        latency = (time.time() - start_time) * 1000
        logger.info(f"Cache HIT for user {user_id} | Latency: {latency:.2f} ms")
        return json.loads(cached_recs)

    latency =(time.time() - start_time) * 1000
    logger.warning(f"Cache MISS for user {user_id} | Latency: {latency:.2f} ms")

    popular_items = get_popular_items()

    redis_client.setex(
        cache_key,
        CACHE_CONFIG["ttl"] // 4,
        json.dumps(popular_items)
    )

    return popular_items

@app.post("/recommendations/batch", response_model=dict)
async def batch_recommendations(user_ids: list[str]):
    results = {}
    for user_id in user_ids:
        results[user_id] = await get_recommendations(user_id)

    return {
        "count": len(results),
        "recommendations": results
    }

@app.get("/")
async def root():
    return {
        "message": "Smart Recommendation Engine API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )