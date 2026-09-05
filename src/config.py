import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR =  BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

DATABASES_URL = os.getenv(
    "DATABASES_URL",
    "postgresql://recommender:recommender123@localhost:5432/recommendations"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "recommendation_system")

MODEL_CONFIG = {
    "no_components": int(os.getenv("MODEL_NO_COMPONENTS", 30)),
    "learning_rate": float(os.getenv("MODEL_LEARNING_RATE", 0.05)),
    "loss": os.getenv("MODEL_LOSS", "warp"),
    "epochs": int(os.getenv("MODEL_EPOCHS", 10)),
}

API_CONFIG = {
    "ttl": int(os.getenv("CACHE_TTL", 3600)),
    "prefix": os.getenv("CACHE_PREFIX", "recs"),
}

CACHE_CONFIG = {
    "ttl": int(os.getenv("CACHE_TTL", 3600)),
    "prefix": os.getenv("CACHE_PREFIX", "recs"),
}

REC_CONFIG = {
    "k": int(os.getenv("RECOMMENDATIONS_K", 10)),
    "fallback": int(os.getenv("FALLBACK_ITEMS", 20)),
}

def get_redis_url() -> str:
    return f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"