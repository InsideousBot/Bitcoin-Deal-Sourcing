# Twitter Scraper Connection Fix

## Problem Summary
The Twitter scraper (twscrape) is failing to connect because **Cloudflare blocks automated login attempts**. The account `smaran_manchala` has been added but cannot log in automatically.

## Solution: Cookie-Based Authentication

Since automated login is blocked, we need to use cookies from a logged-in browser session.

### Option 1: Export Cookies from Browser (RECOMMENDED)

1. **Install a Cookie Exporter Extension:**
   - Chrome: [EditThisCookie](https://chrome.google.com/webstore/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfb)
   - Firefox: [Cookie-Editor](https://addons.mozilla.org/en-US/firefox/addon/cookie-editor/)

2. **Login to Twitter/X manually:**
   - Go to https://x.com
   - Login with: `smaran_manchala`
   - Complete any 2FA/captchas

3. **Export cookies:**
   - Click the extension icon
   - Select "Export" or "Export as JSON"
   - Copy the entire cookie JSON

4. **Save cookies to file:**
   ```bash
   cd /Users/smaran/Desktop/Bitcoin-Deal-Sourcing
   # Paste cookies into a file called cookies.json
   ```

5. **Add account with cookies:**
   ```bash
   source .venv/bin/activate
   twscrape add_accounts --cookies "$(cat cookies.json)" accounts_with_cookies.txt username:password:email:email_password
   ```

### Option 2: Use Playwright (Interactive Login)

Run the helper script that opens a real browser:

```bash
source .venv/bin/activate
python3 setup_twitter_account.py
```

This will open a browser where you can:
- Complete captchas manually
- Handle 2FA
- The script will save cookies automatically

### Option 3: Manual Cookie String

If you can't use extensions, extract cookies manually:

1. Login to https://x.com
2. Open DevTools (F12)
3. Go to Application → Cookies → https://x.com
4. Copy the values for: `auth_token`, `ct0`, `guest_id`
5. Create a cookie file in Netscape format

## Verify Account Status

After adding cookies, check if the account is active:

```bash
source .venv/bin/activate
sqlite3 accounts.db "SELECT username, active, last_used FROM accounts;"
```

You should see: `smaran_manchala|1|<timestamp>` (active=1)

## Test the Pipeline

Once the account is active:

```bash
source .venv/bin/activate
python3 pipeline.py
```

You should see tweets being ingested instead of "No active accounts" errors.

## Current Status

- ✅ Account added: `smaran_manchala`
- ❌ Automated login: Blocked by Cloudflare
- 🔧 Next step: Add cookies from browser session

## Files Modified

- `pipeline.py` - Fixed twscrape initialization logic
- `.env` - Added X_2FA_SECRET field
- `accounts.txt` - Updated format to include email_password
- `accounts.db` - SQLite database where accounts are stored
