# GitHub Setup & Food Tech Cloning Guide

## Part 1: Preparing for GitHub

### Files to Commit to GitHub

✅ **Core Files** (commit these):
- `pipeline.py` - Main pipeline code
- `import_cookies.py` - Cookie import script
- `check_account_status.py` - Diagnostic tool
- `README.md` - Project documentation
- `ARCHITECTURE.md` - Technical documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `.env.example` - Environment template
- `TWITTER_AUTH_FIX.md` - Authentication guide

❌ **Never Commit** (already in .gitignore):
- `.env` - Contains your actual credentials
- `*.db` - Your database files
- `twitter_cookies.json` - Your Twitter session
- `*_export.csv` - Your data exports
- `.venv/` - Virtual environment

### Step-by-Step GitHub Upload

```bash
# 1. Initialize git (if not already done)
cd /Users/smaran/Desktop/Bitcoin-Deal-Sourcing
git init

# 2. Add all files (gitignore will exclude sensitive ones)
git add .

# 3. Commit
git commit -m "Initial commit: Bitcoin deal sourcing pipeline with 7-phase improvements"

# 4. Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/Bitcoin-Deal-Sourcing.git
git branch -M main
git push -u origin main
```

---

## Part 2: Cloning for Food Tech

### Quick Clone & Adapt

```bash
# 1. Clone your Bitcoin repo
cd ~/Desktop
git clone https://github.com/YOUR_USERNAME/Bitcoin-Deal-Sourcing.git Food-Tech-Deal-Sourcing
cd Food-Tech-Deal-Sourcing

# 2. Update remote (optional - if you want separate repo)
git remote remove origin
# Create new repo on GitHub: Food-Tech-Deal-Sourcing
git remote add origin https://github.com/YOUR_USERNAME/Food-Tech-Deal-Sourcing.git

# 3. Create new virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Copy your credentials
cp .env.example .env
# Edit .env with your Twitter credentials (same account works!)
```

### Key Changes for Food Tech

#### 1. Update Keywords in `pipeline.py`

**Find this section** (around line 50):
```python
# ---------------- CONFIG ---------------- #

BITCOIN_KEYWORDS = [
    "bitcoin", "btc", "lightning network", "ordinals", "building on bitcoin",
    "bitvm", "rgb", "fedimint", "covenants", "layers on bitcoin"
]

TRACTION_PHRASES = [
    "beta users", "mainnet soon", "testnet now live",
    "we just shipped", "waitlist open"
]

FOUNDER_KEYWORDS = [
    "founder", "cofounder", "building", "developer",
    "open-source", "ceo", "creator", "maker"
]

BITCOIN_BUILDER_KEYWORDS = [
    "bitcoin", "btc", "lightning", "ordinals", "nostr",
    "fedimint", "taproot", "bitvm", "rollups"
]
```

**Replace with Food Tech keywords**:
```python
# ---------------- CONFIG ---------------- #

FOODTECH_KEYWORDS = [
    "food tech", "foodtech", "restaurant tech", "delivery platform",
    "ghost kitchen", "cloud kitchen", "food delivery", "meal kit",
    "food robotics", "vertical farming", "agtech"
]

TRACTION_PHRASES = [
    "beta users", "soft launch", "now available",
    "we just shipped", "waitlist open", "now serving",
    "expanding to", "new locations"
]

FOUNDER_KEYWORDS = [
    "founder", "cofounder", "building", "developer",
    "open-source", "ceo", "creator", "maker", "chef-founder"
]

FOODTECH_BUILDER_KEYWORDS = [
    "food tech", "restaurant", "delivery", "kitchen",
    "farming", "agriculture", "food safety", "supply chain",
    "meal prep", "catering", "food robotics"
]
```

#### 2. Update Database Name

**Find** (around line 48):
```python
DB_FILE = "deal_sourcing.db"
```

**Change to**:
```python
DB_FILE = "foodtech_sourcing.db"
```

#### 3. Update Variable Names (Optional but Clean)

**Find and replace throughout file**:
- `BITCOIN_KEYWORDS` → `FOODTECH_KEYWORDS`
- `BITCOIN_BUILDER_KEYWORDS` → `FOODTECH_BUILDER_KEYWORDS`
- `bitcoin_builder_score` → `foodtech_builder_score`

**Quick find/replace in terminal**:
```bash
# Backup first
cp pipeline.py pipeline.py.backup

# Replace (macOS)
sed -i '' 's/BITCOIN_KEYWORDS/FOODTECH_KEYWORDS/g' pipeline.py
sed -i '' 's/BITCOIN_BUILDER_KEYWORDS/FOODTECH_BUILDER_KEYWORDS/g' pipeline.py
sed -i '' 's/bitcoin_builder_score/foodtech_builder_score/g' pipeline.py
sed -i '' 's/Bitcoin/Food Tech/g' README.md
```

#### 4. Update Scam Detection Keywords

**Find** `compute_scam_score()` function (around line 280):
```python
spam_keywords = [
    'casino', 'slots', 'double your', '1000%', 'profit guaranteed',
    'signals', 'pump', 'get rich', 'binary options', 'forex',
    'investment opportunity', 'click here', 'dm for', 'telegram group',
    'airdrop', 'giveaway', 'free crypto', 'guaranteed returns'
]
```

**Replace with Food Tech spam patterns**:
```python
spam_keywords = [
    'mlm', 'multi-level', 'join my team', 'be your own boss',
    'work from home', 'unlimited income', 'get rich',
    'investment opportunity', 'click here', 'dm for', 'telegram group',
    'dropshipping course', 'make money online', 'passive income',
    'financial freedom', 'crypto', 'nft'
]
```

#### 5. Update README.md

**Change**:
- Title: "Bitcoin Deal Sourcing" → "Food Tech Deal Sourcing"
- Description: Focus on food tech ecosystem
- Examples: Replace Bitcoin projects with food tech examples

#### 6. Update Airtable Table Name (if using)

**In `.env`**:
```bash
AIRTABLE_TABLE_NAME=FoodTech_Projects
```

---

## Part 3: Running Food Tech Version

```bash
# 1. Activate environment
cd ~/Desktop/Food-Tech-Deal-Sourcing
source .venv/bin/activate

# 2. Import Twitter cookies (same as Bitcoin version)
python3 import_cookies.py

# 3. Run pipeline
python3 pipeline.py
```

**Expected Output**:
```
🚀 Ingesting tweets with twscrape...
Searching for: food tech since:2025-11-23
Searching for: foodtech since:2025-11-23
Searching for: ghost kitchen since:2025-11-23
...
⚠️  Skipping @spammer123 - scam score too high (6.0)
✅ Updated daily top 10 for 2025-11-24
✅ Exported daily leaderboard to daily_leaderboard.csv

🏆 TOP 10 PROJECTS FOR 2025-11-24:
   #1: CloudKitchen (@chef_tech) - Score: 11.2
   #2: FarmBot (@vertical_farm) - Score: 9.8
   ...
```

---

## Part 4: Advanced Customizations

### A. Adjust Scoring Weights

**In `compute_trend_score()` function**:
```python
# Emphasize different factors for food tech
base = (
    0.40 * founder_likelihood +        # Higher weight on founder (restaurants are personal)
    0.15 * foodtech_builder_score +    # Lower weight on keywords
    0.30 * engagement_velocity +       # Higher weight on traction (food is viral)
    0.10 * website_quality_score +
    0.05 * follower_score(followers)   # Lower weight on followers
)
```

### B. Add Food-Specific Product Stages

**In `product_stage_from_text()` function**:
```python
def product_stage_from_text(text: str) -> str:
    lower = text.lower()
    
    # Food tech specific stages
    if any(w in lower for w in ["now open", "grand opening", "multiple locations", "franchising"]):
        return "Scaling"
    if any(w in lower for w in ["soft launch", "pop-up", "first location", "beta"]):
        return "Launched"
    if any(w in lower for w in ["coming soon", "opening soon", "construction", "permits"]):
        return "Pre-launch"
    
    return "Unknown"
```

### C. Add Location Detection

**New helper function**:
```python
import re

def extract_location(bio: str, tweet_text: str) -> str:
    """Extract city/location from bio or tweet"""
    text = f"{bio} {tweet_text}"
    
    # Common patterns
    patterns = [
        r'based in ([A-Z][a-z]+(?: [A-Z][a-z]+)?)',  # "based in San Francisco"
        r'📍\s*([A-Z][a-z]+(?: [A-Z][a-z]+)?)',      # "📍 New York"
        r'located in ([A-Z][a-z]+(?: [A-Z][a-z]+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return "Unknown"
```

**Add to project dict**:
```python
project = {
    ...
    "location": extract_location(bio, tweet["text"]),
    ...
}
```

---

## Part 5: Testing Your Food Tech Version

```bash
# 1. Test with limited tweets first
python3 -c "
from pipeline import BitcoinDealSourcingPipeline
p = BitcoinDealSourcingPipeline()
p.ingest_tweets(hours=24, max_tweets=10)  # Just 10 tweets per keyword
"

# 2. Check results
sqlite3 foodtech_sourcing.db "
    SELECT project_name, trend_score, product_stage, trend_explanation 
    FROM projects 
    ORDER BY trend_score DESC 
    LIMIT 5;
"

# 3. Review daily leaderboard
cat daily_leaderboard.csv
```

---

## Part 6: Quick Reference - File Changes

| File | Change Required | Difficulty |
|------|----------------|------------|
| `pipeline.py` | Update keywords (lines 50-70) | Easy |
| `pipeline.py` | Update DB_FILE name | Easy |
| `pipeline.py` | Update spam keywords | Easy |
| `pipeline.py` | Optional: scoring weights | Medium |
| `README.md` | Update title & description | Easy |
| `.env` | Copy from .env.example | Easy |

---

## Summary

**To create Food Tech version**:
1. ✅ Clone the repo
2. ✅ Replace 4 keyword lists in `pipeline.py`
3. ✅ Update database filename
4. ✅ Update spam detection keywords
5. ✅ Run and test!

**Time estimate**: 15-20 minutes

**Same codebase, different domain** - all the smart filtering, scoring, and daily top 10 tracking works exactly the same! 🚀
