import os 
import json
import mlflow   
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import LabelEncoder

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Olist_LightFM_Hybrid")

BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
MODEL_PATH.mkdir(exist_ok=True)

def load_and_map_data():
    logger.info("Loading processed data...")

    interactions = pd.read_parquet(DATA_PATH / "interactions.parquet")
    item_metadata = pd.read_parquet(DATA_PATH / "item_metadata.parquet")

    user_ids = interactions['customer_id'].unique()
    item_ids = interactions['product_id'].unique()

    user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
    item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}

    user_id_reverse_map = {i: user_id for user_id, i in user_id_map.items()}
    item_id_reverse_map = {i: item_id for item_id, i in item_id_map.items()}

    with open(MODEL_PATH / "user_id_map.json", "w") as f:
        json.dump(user_id_reverse_map, f)
    with open(MODEL_PATH / "item_id_map.json", "w") as f:
        json.dump(item_id_reverse_map, f)

    logger.info(f"Mappings created. Users: {len(user_ids)}, items: {len(item_ids)}")

    return interactions, item_metadata, user_id_map, item_id_map

def build_matrices(interactions, item_metadata, user_id_map, item_id_map):
    logger.info("Building Sparse Matrices...")

    if not isinstance(interactions, pd.DataFrame):
        interactions = interactions.to_pandas()
    if not isinstance(item_metadata, pd.DataFrame):
        item_metadata = item_metadata.to_pandas()

    num_users = len(user_id_map)
    num_items = len(item_id_map)

    user_cats = pd.Categorical(interactions['customer_id'], categories=list(user_id_map.keys()))
    item_cats = pd.Categorical(interactions['product_id'], categories=list(item_id_map.keys()))

    rows = user_cats.codes
    cols = item_cats.codes

    valid_mask = (rows != -1) & (cols != -1)
    rows = rows[valid_mask]
    cols = cols[valid_mask]

    data = np.ones(len(rows), dtype=np.float32)

    interactions_matrix = coo_matrix((data, (rows, cols)), shape=(num_users, num_items))

    item_metadata = item_metadata[item_metadata['product_id'].isin(item_id_map.keys())].copy()

    item_cats_meta = pd.Categorical(item_metadata['product_id'], categories=list(item_id_map.keys()))
    item_metadata['item_idx'] = item_cats_meta.codes

    le = LabelEncoder()

    item_metadata['product_category_name'] = item_metadata['product_category_name'].fillna('unknown')
    item_metadata['category_encoded'] = le.fit_transform(item_metadata['product_category_name'])

    item_features = coo_matrix(
        (np.ones(len(item_metadata)), (item_metadata['item_idx'], item_metadata['category_encoded'])), shape=(num_items, len(le.classes_))
    )

    logger.info(f"Matrices built. Interaction shape: {interactions_matrix.shape}, Item Features: {item_features.shape}")
    return interactions_matrix.tocsr(), item_features.tocsr(), le.classes_
    
def train_and_evaluate(interaction_matrix, item_features):
    logger.info("Starting Model Training...")

    params = {
        "no_components": 32,
        "learning_rate": 0.05,
        "loss": "warp",
        "epochs": 15,
        "num_threads": 4
    }

    with mlflow.start_run(run_name="LightFM_WARP_32Comp"):
        mlflow.log_params(params)

        model = LightFM(no_components=params["no_components"], learning_rate=params["learning_rate"], loss=params["loss"])

        model.fit(
            interaction_matrix,
            item_features=item_features,
            epochs=params["epochs"],
            num_threads=params["num_threads"],
            verbose=True
        )

        logger.info("Evaluating Model...")

        train_precision = precision_at_k(model, interaction_matrix, item_features=item_features, k=10).mean()
        train_recall = recall_at_k(model, interaction_matrix, item_features=item_features, k=10).mean()

        logger.info(f"Train Precision@10: {train_precision:.4f}")
        logger.info(f"Train Recall@10: {train_recall:.4f}")

        mlflow.log_metric("train_precision_at_10", train_precision)
        mlflow.log_metric("train_recall_at_10", train_recall)

        model_path = str(MODEL_PATH / "lightfm_model.pkl")

        with open(MODEL_PATH / "lightfm_model.pkl", "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved locally at {model_path}")

        mlflow.log_artifact(model_path, artifact_path="model")

        logger.info("Model trained and saved successfully")

def main():
    try:
        interactions, item_metadata, user_id_map, item_id_map = load_and_map_data()
        interaction_matrix, item_features, categories = build_matrices(interactions, item_metadata, user_id_map, item_id_map)
        train_and_evaluate(interaction_matrix, item_features)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()