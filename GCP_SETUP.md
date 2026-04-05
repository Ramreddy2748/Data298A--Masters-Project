# GCP Setup & Deployment Guide
## Financial Risk Data Pipeline — TEAM-14, DATA 266, SJSU

---

## PHASE 1 — Run Locally (Start Here)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API keys
```bash
# Get free FRED key: https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY=your_fred_key_here

# Get free NewsAPI key: https://newsapi.org/register
export NEWS_API_KEY=your_news_api_key_here
```

### 3. Run the pipeline
```bash
# Default 5 tickers (AAPL, MSFT, TSLA, NVDA, JPM)
python pipeline.py

# Any tickers you want
python pipeline.py GOOGL META AMZN

# Re-run EDA without re-fetching data (uses Bronze cache)
python pipeline.py --skip-ingest AAPL TSLA NVDA
```

### 4. Outputs
```
data/
├── bronze/          ← Raw CSVs from all 4 sources
├── silver/          ← Cleaned, normalized CSVs
├── gold/            ← Analytics-ready risk score tables
└── reports/         ← 7 PNG visualizations
pipeline.log         ← Full run log
```

---

## PHASE 2 — Set Up GCP

### 1. Create GCP account
- Go to https://console.cloud.google.com
- Sign in with your Google account
- **Free Tier**: $300 credit for 90 days + always-free tier after

### 2. Create or reuse your project
```bash
# You already have: sacred-catfish-488122-u7
# Or create new:
gcloud projects create financial-risk-pipeline --name="Financial Risk Pipeline"
gcloud config set project sacred-catfish-488122-u7
```

### 3. Enable required APIs
```bash
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
```

### 4. Create GCS bucket (Medallion Architecture)
```bash
GCS_BUCKET=financial-risk-pipeline-data

gcloud storage buckets create gs://$GCS_BUCKET \
  --location=us-central1 \
  --uniform-bucket-level-access

# The pipeline creates these prefixes automatically:
# gs://$GCS_BUCKET/bronze/   ← raw data
# gs://$GCS_BUCKET/silver/   ← cleaned data
# gs://$GCS_BUCKET/gold/     ← analytics-ready
```

### 5. Create BigQuery dataset
```bash
bq mk --dataset \
  --location=US \
  --description="Financial risk pipeline Gold layer" \
  sacred-catfish-488122-u7:financial_risk
```

### 6. Authenticate
```bash
gcloud auth application-default login
pip install google-cloud-storage google-cloud-bigquery google-cloud-bigquery-storage pyarrow
```

### 7. Set environment variables and run with GCS
```bash
export GCP_PROJECT_ID=sacred-catfish-488122-u7
export GCS_BUCKET=financial-risk-pipeline-data
export FRED_API_KEY=your_key
export NEWS_API_KEY=your_key

python pipeline.py --gcs AAPL MSFT TSLA NVDA JPM
```

---

## PHASE 3 — Deploy to Cloud Functions (Scheduled)

### 1. Package the pipeline
```bash
zip -r pipeline_package.zip \
  pipeline.py ingest.py clean.py eda.py config.py requirements.txt
```

### 2. Deploy as Cloud Function
```bash
gcloud functions deploy financial-risk-pipeline \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=run_pipeline_gcp \
  --memory=2GB \
  --timeout=540s \
  --set-env-vars="GCP_PROJECT_ID=sacred-catfish-488122-u7,GCS_BUCKET=financial-risk-pipeline-data" \
  --set-secrets="FRED_API_KEY=fred-api-key:latest,NEWS_API_KEY=news-api-key:latest"
```

### 3. Schedule daily runs with Cloud Scheduler
```bash
gcloud scheduler jobs create http financial-risk-daily \
  --location=us-central1 \
  --schedule="0 8 * * *" \
  --uri="https://us-central1-sacred-catfish-488122-u7.cloudfunctions.net/financial-risk-pipeline" \
  --message-body='{"tickers": ["AAPL","MSFT","TSLA","NVDA","JPM"]}' \
  --time-zone="America/Los_Angeles"
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   GCP Project                               │
│              sacred-catfish-488122-u7                       │
│                                                             │
│  ┌─────────────┐    ┌────────────────────────────────────┐  │
│  │ Cloud       │    │     GCS Bucket                     │  │
│  │ Scheduler   │───▶│  financial-risk-pipeline-data      │  │
│  │ (daily 8am) │    │                                    │  │
│  └─────────────┘    │  /bronze/yahoo/   ← raw prices     │  │
│         │           │  /bronze/edgar/   ← financials     │  │
│         ▼           │  /bronze/fred/    ← macro          │  │
│  ┌─────────────┐    │  /bronze/news/    ← headlines      │  │
│  │   Cloud     │    │                                    │  │
│  │  Function   │    │  /silver/prices/  ← cleaned        │  │
│  │  pipeline   │    │  /silver/macro/   ← resampled      │  │
│  └─────────────┘    │  /silver/news/    ← sentiment      │  │
│                     │                                    │  │
│  External APIs:     │  /gold/           ← risk scores    │  │
│  Yahoo Finance  ───▶│                                    │  │
│  SEC EDGAR      ───▶└────────────────────────────────────┘  │
│  FRED API       ───▶         │                              │
│  NewsAPI        ───▶         ▼                              │
│                     ┌────────────────┐                      │
│                     │   BigQuery     │                      │
│                     │financial_risk  │                      │
│                     │  .risk_scores  │                      │
│                     │  .macro        │                      │
│                     │  .news_sent.   │                      │
│                     └────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Cost Estimate (Free Tier)
| Service          | Free Tier              | Our Usage        |
|------------------|------------------------|------------------|
| GCS Storage      | 5 GB/month             | ~50 MB/month     |
| BigQuery Queries | 1 TB/month             | ~1 GB/month      |
| Cloud Functions  | 2M invocations/month   | ~30/month        |
| Cloud Scheduler  | 3 jobs free            | 1 job            |
| **Total Cost**   | **$0/month**           | Within free tier |
