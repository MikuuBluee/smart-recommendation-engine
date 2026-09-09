import os
import json
import pickle
import redis
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from lightfm import LightFM
from scipy.sparse import load_npz

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed"

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def load_assets():
    logger.info("Loading model and assets...")

    with open(MODEL_PATH / "lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODEL_PATH / "user_id_map.json", "r") as f:
        user_map = json.load(f)

    with open(MODEL_PATH / "item_id_map.json", "r") as f:
        item_map = json.load(f)

    item_features = load_npz(str(MODEL_PATH / "item_features.npz"))

    popular_df = pd.read_parquet(DATA_PATH / "popular_items.parquet")
    popular_items = popular_df['product_id'].head(20).tolist()

    logger.info(f"Assets loaded. Model ready, {len(user_map)} users mapped")
    return model, user_map, item_map, popular_items, item_features

def precompute_and_save_to_redis(model, user_map, item_map, popular_items, item_features, k=10):
    logger.info(f"Starting batch prediction for {len(user_map)} users (Top {k})...")

    user_ids_int = [int(uid) for uid in user_map.keys()]
    num_items = len(item_map)
    chunk_size = 1000

    logger.info("Calculating scores and caching to Redis in chunks...")
    pipe = redis_client.pipeline()

    for i in tqdm(range(0, len(user_ids_int), chunk_size), desc="Prosessing chunks"):
        chunk_user_ids = user_ids_int[i : i + chunk_size]
        chunk_scores = []

        for user_idx in chunk_user_ids:
            scores = model.predict(user_idx, np.arange(num_items), item_features=item_features)
            chunk_scores.append(scores)

        chunk_scores_np = np.array(chunk_scores)
        top_k_indices = np.argsort(-chunk_scores_np, axis=1)[:, :k]

        for j, user_idx in enumerate(chunk_user_ids):
            user_id_str = user_map[str(user_idx)]
            recommended_items = [item_map[str(idx)] for idx in top_k_indices[j]]
            pipe.set(f"recs:{user_id_str}", json.dumps(recommended_items))

        pipe.execute()

    redis_client.set("popular_items", json.dumps(popular_items))

    logger.info(f"Batch prediction finished. All recommendations cached in Redis")

def main():
    try:
        model, user_map, item_map, popular_items, item_features = load_assets()
        precompute_and_save_to_redis(model, user_map, item_map, popular_items, item_features, k=10)
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()