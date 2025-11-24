#!/usr/bin/env python3
"""
Main pipeline script for the Bitcoin deal-sourcing engine.

This version fixes:
✔ Twitter ingestion using twscrape (snscrape deprecated)
✔ Async bugs
✔ Ensures twscrape login pool is initialized correctly
✔ Clean orchestration
✔ Uses .env credentials properly
"""

import os
import sqlite3
import datetime
import time
from typing import List
import asyncio

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ---------------- LOAD ENV ---------------- #
load_dotenv()

from twscrape import API

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    import requests
except ImportError:
    requests = None


# ---------------- CONFIG ---------------- #

# Reduced to top 5 keywords to minimize rate limiting with single account
# Phase 5: Expanded with more Bitcoin ecosystem terms
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

WEIGHTS = {
    "founder_likelihood": 0.35,
    "bitcoin_builder_score": 0.25,
    "engagement_velocity": 0.25,
    "website_quality_score": 0.15,
}

DB_FILE = "deal_sourcing.db"
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE_NAME", "Projects")


# ---------------- UTILITIES ---------------- #

def compute_engagement_velocity(tweet) -> float:
    likes = tweet["likes"]
    retweets = tweet["retweets"]
    replies = tweet["replies"]
    hours_since = max((datetime.datetime.utcnow() - tweet["date"]).total_seconds() / 3600, 1e-6)
    return (likes + retweets + replies) / hours_since


def keyword_score(text: str, keywords: List[str]) -> int:
    low = text.lower()
    return sum(1 for kw in keywords if kw in low)


# ============================================================
#          PHASE 4: ENHANCED SCORING
# ============================================================

import math

def follower_score(followers: int) -> float:
    """Logarithmic follower score to avoid over-weighting influencers"""
    return math.log10(followers + 1)


def product_stage_from_text(text: str) -> str:
    """Infer product stage from tweet content"""
    lower = text.lower()
    
    if any(w in lower for w in ["mainnet", "live on mainnet", "production", "now live"]):
        return "Launched"
    if any(w in lower for w in ["beta", "waitlist", "invite-only", "early access"]):
        return "Beta"
    if any(w in lower for w in ["coming soon", "pre-launch", "building", "stealth", "working on"]):
        return "Pre-release"
    
    return "Unknown"


def builder_signal_score(text: str) -> int:
    """Count builder/traction phrases in text"""
    builder_phrases = [
        "we just shipped", "testnet", "mainnet", "beta users", "waitlist",
        "launch", "preseed", "seed round", "dev update", "announcing"
    ]
    lower = text.lower()
    return sum(1 for phrase in builder_phrases if phrase in lower)


def compute_trend_score(
    founder_likelihood: int,
    bitcoin_builder_score: int,
    engagement_velocity: float,
    website_quality_score: float,
    followers: int,
    scam_score: float,
    bio: str,
    tweet_text: str
) -> tuple:
    """
    Enhanced trend score with explanation
    Returns: (score, explanation, product_stage)
    """
    # Add builder signals from tweet content
    builder_signals = builder_signal_score(tweet_text + " " + bio)
    
    # Base score from weighted components
    base = (
        0.30 * founder_likelihood +
        0.20 * (bitcoin_builder_score + builder_signals) +
        0.25 * engagement_velocity +
        0.15 * website_quality_score +
        0.10 * follower_score(followers)
    )
    
    # Penalty for scam signals
    base -= 0.5 * scam_score
    
    reasons = []
    
    # Bonus for small accounts with strong traction (early discovery!)
    if followers < 2000 and engagement_velocity > 5:
        base += 2.0
        reasons.append("small account with strong traction")
    
    # Triple signal bonus: founder + bitcoin + traction
    if founder_likelihood >= 1 and bitcoin_builder_score >= 1 and engagement_velocity > 3:
        base *= 1.2
        reasons.append("founder building on Bitcoin with traction")
    
    # Build explanation
    if engagement_velocity > 5:
        reasons.append(f"high engagement ({engagement_velocity:.1f}/hr)")
    if founder_likelihood > 0:
        reasons.append("founder keywords in bio")
    if bitcoin_builder_score > 0:
        reasons.append("Bitcoin builder")
    if builder_signals > 0:
        reasons.append(f"{builder_signals} traction signals")
    if website_quality_score >= 1:
        reasons.append("has website")
    
    trend_explanation = "; ".join(reasons) if reasons else "organic discovery"
    product_stage = product_stage_from_text(tweet_text + " " + bio)
    
    return max(base, 0.0), trend_explanation, product_stage


def parse_bot_extract(url: str, api_key: str):
    if requests is None or not api_key:
        return None
    try:
        resp = requests.get(
            "https://parse.bot/api/v1/extract",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"url": url},
            timeout=10
        )
        return resp.json() if resp.status_code == 200 else None
    except:
        return None


def embed_texts(texts: List[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts)


def cluster_embeddings(embeddings: np.ndarray, n_clusters=5):
    return KMeans(n_clusters=n_clusters).fit_predict(embeddings)


# ============================================================
#          PHASE 2: EXTRACTION HELPERS
# ============================================================

import re
from typing import Optional

def extract_project_name(bio: str, name: str) -> Optional[str]:
    """Extract project name from bio using patterns"""
    if not bio:
        return None
    
    patterns = [
        r'founder\s*@([A-Za-z0-9_]+)',
        r'co[- ]?founder\s*@([A-Za-z0-9_]+)',
        r'building\s+([A-Za-z0-9_]+)',
        r'working on\s+([A-Za-z0-9_]+)',
        r'creator of\s+([A-Za-z0-9_]+)',
        r'ceo\s*@([A-Za-z0-9_]+)',
        r'founder of\s+([A-Za-z0-9_]+)',
    ]
    
    bio_lower = bio.lower()
    for pattern in patterns:
        match = re.search(pattern, bio_lower, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    
    # If name looks like a brand (single word, capitalized)
    if name and ' ' not in name and len(name) > 0 and name[0].isupper():
        return name
    
    return None


def extract_website(user, tweet_text: str) -> Optional[str]:
    """Extract website with priority logic"""
    # Priority 1: Twitter profile URL
    if hasattr(user, 'url') and user.url:
        return user.url
    
    # Priority 2: Links in bio
    if hasattr(user, 'descriptionLinks') and user.descriptionLinks:
        for link in user.descriptionLinks:
            if hasattr(link, 'url'):
                return link.url
    
    # Priority 3: First URL in tweet
    urls = re.findall(r'https?://[^\s]+', tweet_text)
    if urls:
        return urls[0]
    
    return None


# ============================================================
#          PHASE 3: SCAM DETECTION
# ============================================================

def compute_scam_score(bio: str, website: Optional[str], tweet_text: str) -> float:
    """Detect scam/spam signals"""
    score = 0.0
    
    # Combine all text
    all_text = f"{bio} {tweet_text}".lower()
    
    # Spam keywords
    spam_keywords = [
        'casino', 'slots', 'double your', '1000%', 'profit guaranteed',
        'signals', 'pump', 'get rich', 'binary options', 'forex',
        'investment opportunity', 'click here', 'dm for', 'telegram group',
        'airdrop', 'giveaway', 'free crypto', 'guaranteed returns'
    ]
    
    for keyword in spam_keywords:
        if keyword in all_text:
            score += 2.0
    
    # Check website domain
    if website:
        spam_domains = ['casino', 'bet', 'slots', 'binaryoptions', 'forex', 'gambling']
        for domain in spam_domains:
            if domain in website.lower():
                score += 2.0
    
    return score


# ---------------- SQLITE ---------------- #

def get_connection(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id TEXT PRIMARY KEY,
            user_handle TEXT,
            text TEXT,
            date TIMESTAMP,
            likes INT,
            retweets INT,
            replies INT,
            engagement_velocity REAL,
            relevance_score INT
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            handle TEXT PRIMARY KEY,
            name TEXT,
            bio TEXT,
            followers INT,
            following INT,
            website TEXT,
            account_age_days INT,
            founder_likelihood INT,
            bitcoin_builder_score INT,
            user_activity_score REAL
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            handle TEXT PRIMARY KEY,
            project_name TEXT,
            segment TEXT,
            website TEXT,
            cluster_id INT,
            trend_score REAL,
            followers INT,
            representative_tweet TEXT,
            description TEXT,
            website_quality_score REAL,
            scam_score REAL DEFAULT 0,
            product_stage TEXT,
            trend_explanation TEXT,
            is_top_10_today INTEGER DEFAULT 0,
            last_updated TIMESTAMP
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_top_10 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            handle TEXT,
            rank INTEGER,
            trend_score REAL,
            project_name TEXT,
            reason TEXT,
            UNIQUE(date, handle)
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id INT PRIMARY KEY,
            label TEXT,
            summary TEXT
        );
    """)
    conn.commit()


def upgrade_schema(conn):
    """Adds new columns to existing database if they don't exist"""
    c = conn.cursor()
    
    new_columns = [
        ("projects", "scam_score", "REAL DEFAULT 0"),
        ("projects", "product_stage", "TEXT"),
        ("projects", "trend_explanation", "TEXT"),
        ("projects", "is_top_10_today", "INTEGER DEFAULT 0"),
        ("projects", "last_updated", "TIMESTAMP")
    ]
    
    for table, column, col_type in new_columns:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            print(f"✅ Added {table}.{column}")
        except Exception as e:
            if "duplicate column" in str(e).lower():
                pass  # Column already exists, skip
            else:
                print(f"⚠️  Error adding {table}.{column}: {e}")
    
    conn.commit()


# --------------------------------------------------------
#                    PIPELINE CLASS
# --------------------------------------------------------

class BitcoinDealSourcingPipeline:

    def __init__(self):
        self.conn = get_connection(DB_FILE)
        create_tables(self.conn)
        upgrade_schema(self.conn)  # Phase 1: Migrate existing databases

        self.parse_bot = os.getenv("PARSE_BOT_API_KEY")
        self.airtable_key = os.getenv("AIRTABLE_API_KEY")
        self.airtable_base = os.getenv("AIRTABLE_BASE_ID")
        self.airtable_table = AIRTABLE_TABLE

        # Twitter login info
        self.x_username = os.getenv("X_USERNAME")
        self.x_password = os.getenv("X_PASSWORD")
        self.x_email = os.getenv("X_EMAIL")
        self.x_2fa = os.getenv("X_2FA_SECRET")

        # Shared twscrape API instance
        self.api = API()

    # --------------------------------------------------------
    #             TWITTER INGESTION (TWSCRAPE)
    # --------------------------------------------------------

    async def _init_twscrape(self):
        """Initializes twscrape login pool once."""
        try:
            # Add account to pool
            await self.api.pool.add_account(
                username=self.x_username,
                password=self.x_password,
                email=self.x_email,
                email_password=self.x_password,  # Using same password for email
                mfa_code=self.x_2fa if self.x_2fa else None
            )
            print(f"✅ Added account {self.x_username}")
        except Exception as e:
            print(f"ℹ️  Account may already exist: {e}")
        
        # Login all accounts in the pool
        try:
            await self.api.pool.login_all()
            print("✅ Logged in all accounts")
        except Exception as e:
            print(f"⚠️  Login error: {e}")

    async def _twscrape_search(self, query, limit=50):
        """Search using twscrape and return tweets."""
        results = []
        async for tweet in self.api.search(query, limit=limit):
            results.append(tweet)
        return results
    
    async def _twscrape_search_with_backoff(self, query, limit=50, max_retries=3):
        """Search with automatic backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                print(f"  🔍 Searching: '{query}' (attempt {attempt + 1}/{max_retries})")
                results = []
                async for tweet in self.api.search(query, limit=limit):
                    results.append(tweet)
                print(f"  ✅ Found {len(results)} tweets")
                return results
            except Exception as e:
                error_msg = str(e)
                # Check if it's a rate limit error
                if "No account available" in error_msg or "Next available at" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 120  # 2 minutes
                        print(f"  ⏳ Rate limited, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  ⚠️  Max retries reached, skipping this query")
                        return []
                else:
                    # Non-rate-limit error
                    print(f"  ❌ Error: {error_msg}")
                    return []
        return []

    def ingest_tweets(self, hours=24, max_tweets=50):
        print("🚀 Ingesting tweets with twscrape...")

        since = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)

        async def run_all_queries():
            await self._init_twscrape()

            all_results = []
            all_keywords = BITCOIN_KEYWORDS + TRACTION_PHRASES
            print(f"📊 Searching {len(all_keywords)} keywords with rate-limit backoff...\n")
            
            for idx, kw in enumerate(all_keywords, 1):
                query = f"{kw} since:{since.date()}"
                print(f"[{idx}/{len(all_keywords)}] Keyword: '{kw}'")
                tweets = await self._twscrape_search_with_backoff(query, limit=max_tweets)
                all_results.extend(tweets)
                
                # Small delay between searches to be respectful
                if idx < len(all_keywords):
                    print(f"  💤 Waiting 5s before next search...\n")
                    await asyncio.sleep(5)
            
            print(f"\n✅ Total tweets collected: {len(all_results)}")
            return all_results

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        fetched = loop.run_until_complete(run_all_queries())

        normalized = []
        for t in fetched:
            try:
                normalized.append({
                    "id": str(t.id),
                    "user_handle": f"@{t.user.username}",
                    "user": t.user,  # Store full user object for later processing
                    "text": t.rawContent,
                    "date": t.date.replace(tzinfo=None) if t.date else datetime.datetime.utcnow(),
                    "likes": t.likeCount or 0,
                    "retweets": t.retweetCount or 0,
                    "replies": t.replyCount or 0
                })
            except Exception as e:
                print(f"⚠️  Error processing tweet {t.id}: {e}")
                continue

        for t in normalized:
            t["engagement_velocity"] = compute_engagement_velocity(t)
            t["relevance_score"] = keyword_score(t["text"], BITCOIN_KEYWORDS + TRACTION_PHRASES)
            self._upsert_tweet(t)
            self._process_user_from_tweet(t)

        print(f"✅ Upserted {len(normalized)} tweets.")

    # --------------------------------------------------------
    #                    TWEET DB WRITE
    # --------------------------------------------------------

    def _upsert_tweet(self, t):
        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO tweets VALUES
            (:id, :user_handle, :text, :date, :likes, :retweets, :replies,
             :engagement_velocity, :relevance_score)
        """, t)
        self.conn.commit()

    # --------------------------------------------------------
    #                 USER + PROJECT CREATION
    # --------------------------------------------------------

    def _process_user_from_tweet(self, tweet):
        handle = tweet["user_handle"]
        twitter_user = tweet.get("user")
        
        # Extract real Twitter user data if available
        if twitter_user:
            followers = getattr(twitter_user, 'followersCount', 0) or 0
            following = getattr(twitter_user, 'followingCount', 0) or 0
            bio = getattr(twitter_user, 'rawDescription', '') or ''
            name = getattr(twitter_user, 'displayname', handle.strip('@')) or handle.strip('@')
            
            # Phase 2: Use extract_website helper
            website = extract_website(twitter_user, tweet["text"])
            
            # Calculate account age
            if hasattr(twitter_user, 'created'):
                account_age = datetime.datetime.utcnow() - twitter_user.created.replace(tzinfo=None)
                account_age_days = account_age.days
            else:
                account_age_days = 365  # Default 1 year
        else:
            # Fallback to minimal data if user object not available
            followers = 100
            following = 100
            bio = ""
            name = handle.strip("@")
            website = ""
            account_age_days = 365

        founder_score = keyword_score(bio, FOUNDER_KEYWORDS)
        btc_score = keyword_score(bio, BITCOIN_BUILDER_KEYWORDS)

        user = {
            "handle": handle,
            "name": name,
            "bio": bio,
            "followers": followers,
            "following": following,
            "website": website or "",
            "account_age_days": account_age_days,
            "founder_likelihood": founder_score,
            "bitcoin_builder_score": btc_score,
            "user_activity_score": tweet["engagement_velocity"]
        }

        self._upsert_user(user)
        self._init_project_from_user(user, tweet)

    def _upsert_user(self, user):
        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO users VALUES
            (:handle, :name, :bio, :followers, :following, :website,
             :account_age_days, :founder_likelihood, :bitcoin_builder_score,
             :user_activity_score)
        """, user)
        self.conn.commit()

    def _init_project_from_user(self, user, tweet):
        handle = user["handle"]
        bio = user["bio"] or ""
        
        # Phase 3: Compute scam score
        scam_score = compute_scam_score(bio, user["website"], tweet["text"])
        
        # Phase 3: Filter out scam/low-quality projects
        if scam_score >= 3:
            print(f"⚠️  Skipping {handle} - scam score too high ({scam_score})")
            return
        
        # Skip if no founder signals AND no Bitcoin signals
        if user["founder_likelihood"] == 0 and user["bitcoin_builder_score"] == 0:
            print(f"⚠️  Skipping {handle} - no founder or Bitcoin signals")
            return
        
        # Phase 2: Extract project name using regex patterns
        project_name = extract_project_name(bio, user["name"])
        if not project_name:
            project_name = user["name"] or handle.strip('@')
        
        # Phase 4: Compute enhanced trend score with explanation
        trend_score, trend_explanation, product_stage = compute_trend_score(
            founder_likelihood=user["founder_likelihood"],
            bitcoin_builder_score=user["bitcoin_builder_score"],
            engagement_velocity=user["user_activity_score"],
            website_quality_score=0,  # Will be updated by enrich_projects() if run
            followers=user["followers"],
            scam_score=scam_score,
            bio=bio,
            tweet_text=tweet["text"]
        )

        project = {
            "handle": handle,
            "project_name": project_name,
            "segment": "Unknown",
            "website": user["website"] or "",
            "cluster_id": None,
            "trend_score": trend_score,
            "followers": user["followers"],
            "representative_tweet": tweet["id"],
            "description": bio,
            "website_quality_score": 0,
            "scam_score": scam_score,
            "product_stage": product_stage,
            "trend_explanation": trend_explanation,
            "is_top_10_today": 0,
            "last_updated": datetime.datetime.utcnow()
        }

        self._upsert_project(project)

    def _trend_score(self, user, website_score):
        f = user["founder_likelihood"]
        b = user["bitcoin_builder_score"]
        e = user["user_activity_score"]

        score = (
            WEIGHTS["founder_likelihood"] * f +
            WEIGHTS["bitcoin_builder_score"] * b +
            WEIGHTS["engagement_velocity"] * e +
            WEIGHTS["website_quality_score"] * website_score
        )

        if user["followers"] < 500 and e > 1:
            score += 1

        return round(score, 3)

    def _upsert_project(self, p):
        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO projects VALUES
            (:handle, :project_name, :segment, :website, :cluster_id,
             :trend_score, :followers, :representative_tweet, :description,
             :website_quality_score, :scam_score, :product_stage,
             :trend_explanation, :is_top_10_today, :last_updated)
        """, p)
        self.conn.commit()

    # --------------------------------------------------------
    #                 WEBSITE ENRICHMENT
    # --------------------------------------------------------

    def enrich_projects(self, limit=5):
        if not self.parse_bot:
            print("Skipping enrichment — no Parse.bot key")
            return

        c = self.conn.cursor()
        c.execute("SELECT handle, website, project_name FROM projects ORDER BY trend_score DESC LIMIT ?", (limit,))
        rows = c.fetchall()

        for r in rows:
            res = parse_bot_extract(r["website"], self.parse_bot)
            if not res:
                continue

            desc = res.get("short_description", "")
            cat = res.get("category", "Unknown")

            user = self._get_user(r["handle"])
            website_score = 1
            trend = self._trend_score(user, website_score)

            updated = {
                "handle": r["handle"],
                "project_name": res.get("project_name", r["project_name"]),
                "segment": cat,
                "website": r["website"],
                "cluster_id": None,
                "trend_score": trend,
                "followers": user["followers"],
                "representative_tweet": None,
                "description": desc,
                "website_quality_score": website_score
            }
            self._upsert_project(updated)
            time.sleep(0.5)

    def _get_user(self, handle):
        c = self.conn.cursor()
        c.execute("SELECT * FROM users WHERE handle=?", (handle,))
        row = c.fetchone()
        return dict(row)

    # --------------------------------------------------------
    #                EMBEDDINGS + CLUSTERING
    # --------------------------------------------------------

    def embed_and_cluster(self, n_clusters=5):
        if SentenceTransformer is None or KMeans is None:
            print("Skipping clustering — missing libs")
            return

        c = self.conn.cursor()
        c.execute("SELECT handle, project_name, description, product_stage, segment FROM projects")
        rows = c.fetchall()

        if not rows:
            return

        handles = [r["handle"] for r in rows]
        
        # Phase 6: Better text for embeddings
        texts = []
        for r in rows:
            parts = [
                r['project_name'] or '',
                r['description'] or '',
                r['product_stage'] or '',
                r['segment'] or ''
            ]
            texts.append(" ".join(filter(None, parts)))

        emb = embed_texts(texts)

        labels = cluster_embeddings(emb, n_clusters)

        for cid in set(labels):
            summary = " ; ".join(texts[i] for i in range(len(labels)) if labels[i] == cid)[:500]
            self._upsert_cluster(cid, f"Cluster {cid}", summary)

        for h, cid in zip(handles, labels):
            c.execute("UPDATE projects SET cluster_id=? WHERE handle=?", (int(cid), h))
        self.conn.commit()

        print("Clustering complete.")

    def _upsert_cluster(self, cid, label, summary):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO clusters VALUES (?, ?, ?)", (cid, label, summary))
        self.conn.commit()

    # --------------------------------------------------------
    #                AIRTABLE SYNC
    # --------------------------------------------------------

    def push_to_airtable(self, limit=10):
        if not (self.airtable_key and self.airtable_base):
            print("Skipping Airtable sync — missing credentials")
            return

        c = self.conn.cursor()
        c.execute("""
            SELECT p.project_name, p.handle, p.website, p.segment,
                   p.trend_score, p.followers, p.representative_tweet,
                   p.description, c.label
            FROM projects p
            LEFT JOIN clusters c ON p.cluster_id=c.id
            ORDER BY p.trend_score DESC
            LIMIT ?
        """, (limit,))
        rows = c.fetchall()

        headers = {
            "Authorization": f"Bearer {self.airtable_key}",
            "Content-Type": "application/json"
        }
        url = f"https://api.airtable.com/v0/{self.airtable_base}/{self.airtable_table}"

        for r in rows:
            body = {
                "fields": {
                    "Project Name": r["project_name"],
                    "Founder Handle": r["handle"],
                    "Website": r["website"],
                    "Segment": r["segment"],
                    "Cluster Label": r["label"],
                    "Trend Score": r["trend_score"],
                    "Followers": r["followers"],
                    "Representative Tweet": r["representative_tweet"],
                    "Description": r["description"],
                    "Status": "New"
                }
            }
            try:
                requests.post(url, json=body, headers=headers, timeout=10)
            except:
                pass

    # --------------------------------------------------------
    #              PHASE 7: DAILY TOP 10 TRACKER
    #--------------------------------------------------------

    def update_daily_top_10(self):
        """Track today's top 10 projects"""
        from datetime import date
        today = date.today()
        c = self.conn.cursor()
        
        # Reset is_top_10_today flag
        c.execute("UPDATE projects SET is_top_10_today = 0")
        
        # Get top 10 by trend_score with filters
        c.execute("""
            SELECT handle, project_name, trend_score, trend_explanation
            FROM projects
            WHERE trend_score > 3 AND scam_score < 2
            ORDER BY trend_score DESC
            LIMIT 10
        """)
        
        top_10 = c.fetchall()
        
        for rank, row in enumerate(top_10, 1):
            # Mark as top 10 today
            c.execute("UPDATE projects SET is_top_10_today = 1 WHERE handle = ?", (row['handle'],))
            
            # Record in daily_top_10 table
            c.execute("""
                INSERT OR REPLACE INTO daily_top_10 (date, handle, rank, trend_score, project_name, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (today, row['handle'], rank, row['trend_score'], row['project_name'], row['trend_explanation']))
        
        self.conn.commit()
        print(f"✅ Updated daily top 10 for {today}")
        return top_10


    def export_daily_leaderboard(self, output_file="daily_leaderboard.csv"):
        """Export today's top projects"""
        from datetime import date
        import csv
        
        today = date.today()
        c = self.conn.cursor()
        
        c.execute("""
            SELECT 
                rank,
                project_name,
                handle,
                trend_score,
                reason
            FROM daily_top_10
            WHERE date = ?
            ORDER BY rank
        """, (today,))
        
        rows = c.fetchall()
        
        if not rows:
            print("⚠️  No top 10 data for today. Run update_daily_top_10() first.")
            return
        
        # Also get additional project details
        detailed_rows = []
        for row in rows:
            c.execute("""
                SELECT followers, product_stage, website, segment
                FROM projects
                WHERE handle = ?
            """, (row['handle'],))
            details = c.fetchone()
            
            detailed_rows.append({
                'rank': row['rank'],
                'project': row['project_name'],
                'handle': row['handle'],
                'score': f"{row['trend_score']:.2f}",
                'followers': details['followers'] if details else 0,
                'stage': details['product_stage'] if details else 'Unknown',
                'segment': details['segment'] if details else 'Unknown',
                'website': details['website'] if details else 'N/A',
                'why_top_10': row['reason']
            })
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['rank', 'project', 'handle', 'score', 'followers', 'stage', 'segment', 'website', 'why_top_10'])
            writer.writeheader()
            writer.writerows(detailed_rows)
        
        print(f"✅ Exported daily leaderboard to {output_file}")
        print(f"\n🏆 TOP 10 PROJECTS FOR {today}:")
        for row in detailed_rows[:10]:
            print(f"   #{row['rank']}: {row['project']} ({row['handle']}) - Score: {row['score']}")


    # --------------------------------------------------------
    #                   RUN PIPELINE
    # --------------------------------------------------------

    def run(self):
        self.ingest_tweets()
        self.enrich_projects()
        self.embed_and_cluster()
        self.update_daily_top_10()  # Phase 7: Track top 10
        self.export_daily_leaderboard()  # Phase 7: Export CSV
        self.push_to_airtable()


if __name__ == "__main__":
    BitcoinDealSourcingPipeline().run()
