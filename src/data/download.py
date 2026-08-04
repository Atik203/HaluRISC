"""
Data acquisition script for HaluRISC.
Downloads HaluEval QA dataset (qa_data.json - JSONL format) from GitHub.
"""

import os
import json
import logging
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
OUTPUT_DIR = os.path.join("data", "raw", "halueval")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qa_data.json")

def download_halueval_qa():
    """Download HaluEval qa_data.json if not already present."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.info(f"Downloading HaluEval QA dataset from {HALUEVAL_QA_URL}...")
    try:
        urllib.request.urlretrieve(HALUEVAL_QA_URL, OUTPUT_FILE)
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        logging.info(f"Successfully downloaded HaluEval QA dataset ({size_mb:.2f} MB) to {OUTPUT_FILE}")
        
        # Verify JSONL validity
        count = 0
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    count += 1
        logging.info(f"Verified dataset: {count} JSONL items loaded successfully.")
        return OUTPUT_FILE
    except Exception as e:
        logging.error(f"Failed to download or verify HaluEval QA dataset: {e}")
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        raise

if __name__ == "__main__":
    download_halueval_qa()
