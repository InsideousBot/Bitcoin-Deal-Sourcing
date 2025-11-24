# Bitcoin Deal Sourcing Platform

This repository implements a lightweight, fully-scriptable deal-sourcing engine designed to identify rising founders and emerging startups in the Bitcoin ecosystem.  
It uses:

- Python  
- snscrape or twscrape  
- SQLite  
- FAISS or Chroma  
- sentence-transformers  
- parse.bot  
- Airtable API  
- GitHub Actions  

The pipeline returns a daily ranked list of founders/startups building in Bitcoin.

---

## Features

### ✔ Twitter/X Ingestion  
Searches for public Bitcoin-related tweets using:
- bitcoin, btc, lightning, lightning network  
- ordinals, taproot assets, fedimint, bitvm  
- “we just shipped”, “mainnet soon”, “waitlist open”

### ✔ Founder + Project Scoring  
Scores users based on:
- Bio keywords (founder, building, developer…)  
- Bitcoin ecosystem relevance  
- Engagement velocity  
- Website presence  

### ✔ Website Enrichment via parse.bot  
Parses startup landing pages for:
- Project name  
- Tagline  
- Description  
- Waitlist/CTA  

### ✔ Clustering & Embeddings  
Uses `sentence-transformers` + FAISS to cluster startups into themes like:
- Lightning infrastructure  
- Bitcoin L2  
- Ordinals tooling  
- Wallets/custody systems  

### ✔ Trend Scoring  
Final score =  
`0.35 founder + 0.25 bitcoin relevance + 0.25 engagement + 0.15 website quality`.

### ✔ Airtable Sync  
Pushes top projects into your **Bitcoin Deal Sourcing** Airtable base.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Running the Pipeline

```bash
source .venv/bin/activate
python3 pipeline.py
```

### Performance Expectations (Single Twitter Account)

With one authenticated Twitter account:
- **Collection Rate**: ~600-1,000 tweets per run
- **Duration**: ~5-10 minutes per run  
- **Frequency**: Run every 6-12 hours for best results
- **Rate Limits**: Automatic backoff handles Twitter rate limiting

> **Note**: To scale up, add 5-10 Twitter accounts to the pool or integrate Apify (see `implementation_plan.md`)

### Accessing Your Data

Your scraped data is stored in `deal_sourcing.db` (SQLite database).

**View top projects**:
```bash
sqlite3 deal_sourcing.db "SELECT project_name, trend_score, followers FROM projects ORDER BY trend_score DESC LIMIT 10;"
```

**Export to CSV**:
```bash
sqlite3 -header -csv deal_sourcing.db "SELECT * FROM projects;" > projects.csv
```

**Using Python**:
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('deal_sourcing.db')
projects = pd.read_sql_query("SELECT * FROM projects ORDER BY trend_score DESC", conn)
print(projects.head(20))
```

---

## Database Schema

- **tweets** - Raw tweet data with engagement metrics
- **users** - User profiles with founder/builder scores  
- **projects** - Scored projects with clustering
- **clusters** - AI-generated project categories
