import os
import glob
from pathlib import Path


########################### Artifacts #########################

# Saveing Financial data fetched from yfinance
FIN_DATA="artifacts/all_fundamental_data.csv"

# ---  Load/Save Threshold Config for fundamentals & Technicals---
CONFIG_FILE = "artifacts/fundamental_values.json"

# For Embedding the sentence
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG DB Path
DB_FAISS_PATH = "vectorstore/db_faiss_finance"

# list of stocks 
STOCK_LIST="artifacts/stock_list.json"

# huggingface Model 
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# Logs saved
LOG_DIR="logs"

funda_agent_file='artifacts/fundamental_agent_data/funda_agent_data.json'

golden_file='artifacts/golden_baseline/golden_data.json'

golden_file_tech='artifacts/golden_baseline/golden_tech_data.json'

tech_agent_file="artifacts/tech_agent_data/tech_agent_data.json"

validation_report="agent_validation_report/fundamental_validation_report.csv"

dash_path="Agent_validation_report/fundamental_validation_report.csv"
