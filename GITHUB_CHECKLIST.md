# GitHub Commit Checklist

## ✅ Files Ready to Commit

### Core Application Files
- [x] `pipeline.py` - Main pipeline with all 7 phases
- [x] `import_cookies.py` - Twitter cookie authentication
- [x] `check_account_status.py` - Account diagnostics
- [x] `requirements.txt` - Python dependencies

### Documentation
- [x] `README.md` - Project overview and setup
- [x] `ARCHITECTURE.md` - Technical deep dive
- [x] `TWITTER_AUTH_FIX.md` - Authentication troubleshooting
- [x] `GITHUB_AND_CLONING_GUIDE.md` - This guide + Food Tech instructions

### Configuration
- [x] `.gitignore` - Protects sensitive files
- [x] `.env.example` - Environment template

## ❌ Files NOT to Commit (Protected by .gitignore)

- `.env` - Your actual credentials
- `*.db` - Database files (deal_sourcing.db, accounts.db)
- `twitter_cookies.json` - Your Twitter session
- `*_export.csv` - Your data exports
- `daily_leaderboard.csv` - Generated reports
- `.venv/` - Virtual environment

## Quick Commands

```bash
# Initialize and push to GitHub
cd /Users/smaran/Desktop/Bitcoin-Deal-Sourcing
git init
git add .
git commit -m "Initial commit: Bitcoin deal sourcing with intelligent filtering"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Bitcoin-Deal-Sourcing.git
git push -u origin main
```

## For Food Tech Clone

See detailed instructions in `GITHUB_AND_CLONING_GUIDE.md`

**Quick version**:
1. Clone repo → `Food-Tech-Deal-Sourcing`
2. Update 4 keyword lists in `pipeline.py` (lines 50-70)
3. Change `DB_FILE = "foodtech_sourcing.db"`
4. Run: `python3 pipeline.py`

Done! 🚀
