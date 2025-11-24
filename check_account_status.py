#!/usr/bin/env python3
"""
Diagnostic script to check twscrape account status and rate limits
"""
import asyncio
from twscrape import API
import sqlite3
from datetime import datetime

async def check_status():
    api = API()
    
    print("=" * 60)
    print("TWSCRAPE ACCOUNT STATUS CHECK")
    print("=" * 60)
    
    # Get accounts from pool
    accounts = await api.pool.accounts_info()
    
    if not accounts:
        print("\n❌ No accounts found in pool!")
        return
    
    print(f"\n✅ Found {len(accounts)} account(s)")
    
    for acc in accounts:
        # Handle both dict and object formats
        username = acc.get('username') if isinstance(acc, dict) else getattr(acc, 'username', 'Unknown')
        active = acc.get('active') if isinstance(acc, dict) else getattr(acc, 'active', False)
        last_used = acc.get('last_used') if isinstance(acc, dict) else getattr(acc, 'last_used', 'Never')
        
        print(f"\n📊 Account: {username}")
        print(f"   Status: {'✅ Active' if active else '❌ Inactive'}")
        print(f"   Last Used: {last_used}")
        
        # Check for locks (rate limits)
        locks = acc.get('locks') if isinstance(acc, dict) else getattr(acc, 'locks', {})
        if locks:
            print(f"   🔒 Rate Limits:")
            for endpoint, lock_time in locks.items():
                if lock_time:
                    print(f"      - {endpoint}: locked until {lock_time}")
        else:
            print(f"   ✅ No active rate limits")
    
    # Try a test search
    print("\n" + "=" * 60)
    print("TESTING SEARCH FUNCTIONALITY")
    print("=" * 60)
    
    try:
        print("\n🔍 Attempting to search for 'bitcoin' (limit 3)...")
        count = 0
        async for tweet in api.search("bitcoin", limit=3):
            count += 1
            print(f"\n✅ Tweet {count}:")
            print(f"   User: @{tweet.user.username}")
            print(f"   Text: {tweet.rawContent[:100]}...")
            print(f"   Date: {tweet.date}")
            print(f"   Engagement: {tweet.likeCount} likes, {tweet.retweetCount} retweets")
        
        if count == 0:
            print("\n⚠️  Search returned 0 results - possible rate limiting or account issue")
        else:
            print(f"\n✅ Successfully fetched {count} tweets!")
            
    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        print(f"\nThis could indicate:")
        print("  1. Rate limiting is active")
        print("  2. Account is not properly authenticated")
        print("  3. Twitter API is blocking requests")

if __name__ == "__main__":
    asyncio.run(check_status())
