#  Smart Recommendation Engine

Sistem rekomendasi e-commerce hibrida (Hybrid Recommendation System) yang dibangun dengan arsitektur MLOps modern. Project ini menggunakan dataset Olist untuk memprediksi produk yang paling relevan bagi pengguna, mengatasi masalah *cold-start*, dan menyajikan rekomendasi dengan latensi rendah.

## 🛠️ Tech Stack

- **Data Processing:** Python, Polars, Pandas
- **Machine Learning:** LightFM (Hybrid Collaborative Filtering)
- **API & Serving:** FastAPI, Uvicorn
- **Caching & Database:** Redis, PostgreSQL (pgvector)
- **MLOps & Tracking:** MLflow, Docker, GitHub Actions
- **Testing:** Pytest

## ✨ Key Features

- **Hybrid Model:** Menggabungkan data interaksi user dan metadata produk untuk mengatasi *cold-start problem*.
- **Low Latency Serving:** Menggunakan Redis untuk caching rekomendasi, mencapai latensi < 100ms.
- **MLOps Ready:** Tracking eksperimen model menggunakan MLflow dan containerized dengan Docker.
- **Batch Pre-computation:** Menghitung rekomendasi secara massal untuk efisiensi resource.

## 📁 Project Structure

```text
smart-recommendation-engine/
├── .github/workflows/      # CI/CD Pipeline
├── data/                   # Dataset (Raw & Processed)
├── docker/                 # Docker configurations
├── notebooks/              # EDA & Eksperimen Jupyter
├── src/
│   ├── data/               # ETL & Data Pipelines
│   ├── models/             # Training & Evaluasi Model
│   ├── serving/            # FastAPI Application
│   └── utils/              # Helper functions & Config
├── docker-compose.yml      # Orchestration
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (Opsional, jika run di luar Docker)

### Installation & Running

1. Clone repository ini:
```bash
git clone https://github.com/USERNAME_GITHUB_KAMU/smart-recommendation-engine.git
cd smart-recommendation-engine
```

2. Setup environment variables:
```bash
cp .env.example .env
```

3. Jalankan semua services (Redis, Postgres, MLflow, API) menggunakan Docker:
```bash
docker-compose up -d --build
```

4. Akses API Documentation (Swagger UI):
Buka browser dan kunjungi: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📊 Dataset

Project ini menggunakan **Brazilian E-Commerce Public Dataset by Olist** dari Kaggle. 
Untuk menjalankan pipeline, download dataset dan letakkan file CSV-nya di folder `data/raw/`.

## 📡 API Endpoints

- `GET /health` - Health check service & Redis connection.
- `GET /recommend/{user_id}` - Mendapatkan top-K rekomendasi untuk user tertentu.
- `POST /recommend/batch` - Mendapatkan rekomendasi untuk multiple users.