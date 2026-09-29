# Local Environment & Setup Guide

## 1. Prerequisites
- **Python**: Version 3.10+ (Tested on Python 3.13)
- **Node.js**: Version 18+ (Tested on Node 22.16)
- **npm**: Version 9+ (Tested on npm 11.4)

## 2. Python Environment & Dependencies Installation
```bash
# Install required Python packages
pip install -r requirements.txt
```

## 3. Data Processing & Model Pipeline Execution
```bash
# 1. Dataset Inspection
python3 scripts/inspect_data.py

# 2. Data Cleaning & Preprocessing
python3 scripts/preprocess.py

# 3. XGBoost Model Training & Evaluation
python3 scripts/train_xgboost.py
python3 scripts/evaluate_xgboost.py

# 4. News Sentiment Pipeline (Optional if raw news CSV present)
python3 scripts/process_news.py
python3 scripts/run_finbert.py

# 5. Temporal Sequence & PyTorch LSTM Pipeline
python3 scripts/prepare_sequences.py
python3 scripts/train_lstm.py
python3 scripts/evaluate_lstm.py

# 6. Ablation Study & Report Generation
python3 scripts/run_ablation_study.py

# 7. Run Integration Test Suite
python3 tests/test_pipeline.py -v
```

## 4. Running Backend Server
```bash
# Start FastAPI backend server on port 8000
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## 5. Running Next.js Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies (if not already done)
npm install

# Start Next.js development server
npm run dev
```
Access the interactive dashboard in your browser at `http://localhost:3000`.
