from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis
import json
import time
import os
from typing import List
from loguru import logger

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")

redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    logger.info("Connected to Redis Cache")
    yield
    redis_client.close()
    logger.info("Disconnected from Redis")

app = FastAPI(
    title="Smart Recommendation Engine API",
    description="Hybrid Recommendation System (LightFM) with Redis Caching",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
async def health_check():
    try:
        is_connected = redis_client.ping()
        cached_users = redis_client.dbsize()
        return {
            "status": "healthy",
            "redis": "connected" if is_connected else "disconnected",
            "cached_recommendations": cached_users
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/recommend/{user_id}", response_model=List[str])
async def get_recommendations(user_id: str):
    start_time = time.time()
    cache_key = f"recs:{user_id}"

    cached_recs = redis_client.get(cache_key)

    if cached_recs:
        latency = (time.time() - start_time) * 1000
        logger.info(f"Cache HIT for user {user_id} | Latency: {latency:.2f}ms")
        return json.loads(cached_recs)

    popular_items_str = redis_client.get("popular_items")
    popular_items = json.loads(popular_items_str) if popular_items_str else []

    latency = (time.time() - start_time) * 1000
    logger.warning(f"Cache MISS (New User) for {user_id} | Latency: {latency:.2f}ms")

    return popular_items

@app.get("/")
async def root():
    return {
        "message": "Smart Recommendation Engine API is running.",
        "swaagger_docs": "/docs"
    }