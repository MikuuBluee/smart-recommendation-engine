# 🚀 Smart Recommendation Engine

Sistem rekomendasi e-commerce **production-grade** yang dibangun dengan arsitektur MLOps modern. Menggunakan dataset Olist (Brazilian E-Commerce) untuk memprediksi produk yang paling relevan bagi pengguna, mengatasi masalah *cold-start*, dan menyajikan rekomendasi dengan latensi **< 5ms**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Key Features

-  **Hybrid Model**: LightFM dengan collaborative filtering + content-based features
-  **Ultra-Low Latency**: Redis caching untuk response time < 5ms
- 🔄 **Batch Pre-computation**: Pre-compute 96K+ user recommendations
- ️ **Cold-Start Handling**: Fallback ke popular items untuk user baru
- 📊 **MLflow Tracking**: Experiment tracking & model versioning
- 🐳 **Dockerized**: Full containerized dengan docker-compose
- 🧪 **CI/CD Ready**: GitHub Actions untuk automated testing & deployment

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   User Request  │────▶│   FastAPI    │────▶│    Redis    │
│   (HTTP/REST)   │     │   (Serving)  │     │   (Cache)   │
└─────────────────     └──────────────┘     └─────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │  LightFM     │
                        │  (Model)     │
                        └──────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │   MLflow     │
                        │  (Tracking)  │
                        └──────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **ML Model** | LightFM (Hybrid Collaborative Filtering) |
| **Cache** | Redis 7 |
| **Database** | PostgreSQL 16 + pgvector |
| **ML Tracking** | MLflow 2.9 |
| **Data Processing** | Polars + Pandas |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest |

---

## 📁 Project Structure

```
smart-recommendation-engine/
├── .github/workflows/      # CI/CD Pipeline (GitHub Actions)
├── data/
│   ├── raw/                # Raw CSV files (Olist dataset)
│   └── processed/          # Parquet feature store
├── docker/                 # Docker configurations
├── models/                 # Trained models & mappings
├── notebooks/              # EDA & experimentation
├── src/
│   ├── data/               # ETL & batch processing
│   │   ├── etl_pipeline.py
│   │   └── batch_predictor.py
│   ├── models/             # Model training
│   │   └── train_lightfm.py
│   ├── serving/            # FastAPI application
│   │   └── main.py
│   └── utils/              # Helper functions
├── tests/                  # Unit tests
├── docker-compose.yml      # Service orchestration
├── Dockerfile              # API container
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (optional, for local development)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-recommendation-engine.git
cd smart-recommendation-engine
```

### 2. Setup Environment
```bash
cp .env.example .env
```

### 3. Start All Services
```bash
docker-compose up -d --build
```

Services yang akan berjalan:
- **API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Redis**: localhost:6379
- **PostgreSQL**: localhost:5432

### 4. Run ETL Pipeline
```bash
docker-compose exec api python src/data/etl_pipeline.py
```

### 5. Train Model
```bash
docker-compose exec api python src/models/train_lightfm.py
```

### 6. Pre-compute Recommendations
```bash
docker-compose exec api python src/data/batch_predictor.py
```

### 7. Test API
Buka http://localhost:8000/docs untuk interactive API documentation.

---

##  API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "redis": "connected",
  "cached_recommendations": 96478
}
```

### `GET /recommend/{user_id}`
Get Top-10 recommendations for a user.

**Parameters:**
- `user_id` (string): Unique user identifier

**Response:**
```json
[
  "product_id_1",
  "product_id_2",
  ...
  "product_id_10"
]
```

**Performance:**
- Cache HIT (existing user): **< 5ms**
- Cache MISS (new user): **< 10ms** (fallback to popular items)

---

## 📊 Dataset

Project ini menggunakan **Brazilian E-Commerce Public Dataset by Olist** dari Kaggle.

**Dataset Stats:**
-  100,000+ orders
- 👥 96,478 unique customers
- 🛍️ 32,216 unique products
- 📅 Period: 2016-09 to 2018-08

Download: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

---

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

##  Docker Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Restart specific service
docker-compose restart api
```

---

## 📈 MLflow Tracking

Semua eksperimen model tercatat di MLflow UI:
- **URL**: http://localhost:5000
- **Experiment**: Olist_LightFM_Hybrid
- **Metrics**: Precision@10, Recall@10
- **Artifacts**: Model files, feature matrices

---

## 🚢 Deployment

### Option 1: Deploy to VPS
```bash
# SSH ke server
ssh user@your-vps-ip

# Clone & run
git clone <repo-url>
docker-compose up -d
```

### Option 2: Deploy to Railway/Render
1. Connect GitHub repository
2. Set environment variables
3. Deploy automatically

### Option 3: Deploy to AWS ECS
```bash
aws ecs create-cluster --cluster-name recommender-cluster
# ... (configure ECS service)
```

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## 👤 Author

Dicky Widya Pergodi

**Connect:**
- GitHub: https://github.com/MikuuBluee
- LinkedIn: www.linkedin.com/in/dicky-widya-pergodi
```

---