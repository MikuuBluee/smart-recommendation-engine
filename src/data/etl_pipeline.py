import pandas as pd
import numpy as np
from pathlib import Path
import polars as pl
from loguru import logger
from datetime import datetime
import os

logger.add("logs/etl_pipeline_{time}.log", rotation="1 day", retention="7 days")

class OlistETLPipeline:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path(processed_data_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        logger.info("Olist ETL Pipeline initialized")
        logger.info(f"Raw data path: {self.raw_path}")
        logger.info(f"Processed data path: {self.processed_path}")

    def load_data(self) -> dict:
        logger.info("Loading raw CSV files...")

        datasets = {}
        csv_file = {
            'customers': 'olist_customers_dataset.csv',
            'geolocation': 'olist_geolocation_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'order_payments': 'olist_order_payments_dataset.csv',
            'order_reviews': 'olist_order_reviews_dataset.csv',
            'orders': 'olist_orders_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'category_translation': 'product_category_name_translation.csv'
        }

        for name, filename in csv_file.items():
            filepath = self.raw_path / filename
            if filepath.exists():
                datasets[name] = pl.read_csv(filepath, try_parse_dates=True)
                logger.info(f"Loaded {name}: {datasets[name].shape[0]} rows, {datasets[name].shape[1]} columns")
            else:
                logger.warning(f"File not found: {filename}")

        return datasets

    def clean_orders(self, orders_df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Cleaning order data...")

        orders_clean = orders_df.filter(
            pl.col("order_status") == "delivered"
        )

        orders_clean = orders_clean.drop_nulls(subset=["order_purchase_timestamp"])

        logger.info(f"Cleaned orders: {orders_clean.shape[0]} delivered orders")
        return orders_clean

    def create_interaction_matrix(self, datasets: dict) -> pl.DataFrame:
        logger.info("Creating user-item interaction matrix...")

        orders = datasets['orders']
        order_items = datasets['order_items']
        products = datasets['products']
        customers = datasets['customers']

        interactions = order_items.join(orders, on="order_id", how="inner")
        interactions = interactions.join(customers, on="customer_id", how="inner")
        interactions = interactions.join(products, on="product_id", how="inner")

        interactions = interactions.select([
            "customer_id",
            "product_id",
            "order_id",
            "order_purchase_timestamp",
            "price",
            "freight_value",
            "product_category_name"
        ])

        interactions = interactions.sort("order_purchase_timestamp")

        logger.info(f"Interaction matrix created: {interactions.shape[0]} interactions")
        logger.info(f"Unique customers: {interactions['customer_id'].n_unique()}")
        logger.info(f"Unique products: {interactions['product_id'].n_unique()}")

        return interactions

    def create_item_metadata(self, datasets: dict) -> pl.DataFrame:
        logger.info("Creating item metadata...")

        order_items = datasets['order_items']
        products = datasets['products']

        item_metadata = order_items.join(products, on="product_id", how="left").select([
            "product_id",
            "product_category_name",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm"
        ])

        item_metadata = item_metadata.unique(subset=["product_id"])

        item_metadata = item_metadata.fill_null(0)

        logger.info(f"Item metadata created: {item_metadata.shape[0]} unique products")
        return item_metadata

    def create_user_metadata(self, datasets: dict) -> pl.DataFrame:
        logger.info("Creating user metadata...")

        customers = datasets['customers']
        orders = datasets['orders']

        user_metadata = customers.join(orders, on="customer_id", how="left").select([
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
            "customer_zip_code_prefix"
        ])

        user_metadata = user_metadata.unique(subset=["customer_id"])

        logger.info(f"User metadata created: {user_metadata.shape[0]} unique customers")
        return user_metadata

    def create_popular_items(self, interactions: pl.DataFrame) -> pl.DataFrame:
        logger.info("Calculating popular items...")

        popular_items = interactions.group_by("product_id").agg([
            pl.col("order_id").count().alias("total_orders"),
            pl.col("price").sum().alias("total_revenue"),
            pl.col("price").mean().alias("avg_price")
        ])

        popular = popular.sort("total_orders", descending=True)

        logger.info(f"Top 10 popular items calculated")
        return popular

    def save_feature_store(self, interactions: pl.DataFrame, item_metadata: pl.DataFrame, user_metadata: pl.DataFrame, popular_items: pl.DataFrame):
        logger.info("Saving feature store to parquet files...")

        interactions_path = self.processed_path / "interactions.parquet"
        interactions.write_parquet(interactions_path)
        logger.info(f"Saved interactions: {interactions_path}")

        item_metadata_path = self.processed_path / "item_metadata.parquet"
        item_metadata.write_parquet(item_metadata_path)
        logger.info(f"Saved item metadata: {item_metadata_path}")

        user_metadata_path = self.processed_path / "user_metadata.parquet"
        user_metadata.write_parquet(user_metadata_path)
        logger.info(f"Saved user metadata: {user_metadata_path}")

        popular_path = self.processed_path / "popular_items.parquet"
        popular_items.write_parquet(popular_path)
        logger.info(f"Saved popular items: {popular_path}")

        interactions.write_csv(self.processed_path / "interactions.csv")
        item_metadata.write_csv(self.processed_path / "item_metadata.csv")
        user_metadata.write_csv(self.processed_path / "user_metadata.csv")
        popular_items.write_csv(self.processed_path / "popular_items.csv")

        logger.info("Also saved as CSV files for inspection")

    def run(self):
        logger.info("="*60)
        logger.info("STARTING OLIST ETL PIPELINE")
        logger.info("="*60)

        start_time = datetime.now()

        try:
            datasets = self.load_data()

            if not datasets:
                logger.error("No datasets loaded. Check if CSV files exist in raw folder")
                return

            datasets['orders'] = self.clean_orders(datasets['orders'])

            interactions = self.create_interaction_matrix(datasets)
            item_metadata = self.create_item_metadata(datasets)
            user_metadata = self.create_user_metadata(datasets)
            popular_items = self.create_popular_items(interactions)

            self.save_feature_store(interactions, item_metadata, user_metadata, popular_items)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info("="*60)
            logger.info(f"ETL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {elapsed:.2f} seconds")
            logger.info("="*60)

            logger.info("\n DATA SUMMARY:")
            logger.info(f"Total interactions: {interactions.shape[0]}")
            logger.info(f"Unique customers: {interactions['customer_id'].n_unique()}")
            logger.info(f"Unique products: {interactions['product_id'].n_unique()}")
            logger.info(f"Date range: {interactions['order_purchase_timestamp'].min()} to {interactions['order_purchase_timestamp'].max()}")
            logger.info(f"Sparsity: {1 - (interactions.shape[0] / (interactions['customer_id'].n_unique() * interactions['product_id'].n_unique())):.4f}")

        except Exception as e:
            logger.error(f" ETL Pipeline failed: {str(e)}")
            raise e

def main():
    BASE_DIR = Path(__file__).parent.parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"

    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    pipeline = OlistETLPipeline(
        raw_data_path=str(RAW_DATA_PATH),
        processed_data_path=str(PROCESSED_DATA_PATH)
    )

    pipeline.run()

if __name__ == "__main__":
    main()
    