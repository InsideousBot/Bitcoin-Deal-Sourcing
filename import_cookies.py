#!/usr/bin/env python3
"""
Import Twitter cookies into twscrape to activate the account
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from twscrape import API

load_dotenv()

async def import_cookies():
    api = API()
    
    # Load cookies from JSON file
    with open('twitter_cookies.json', 'r') as f:
        cookies_json = json.load(f)
    
    # Convert to cookie string format that twscrape expects
    cookie_str = ""
    for cookie in cookies_json:
        cookie_str += f"{cookie['name']}={cookie['value']}; "
    cookie_str = cookie_str.strip()
    
    print(f"🍪 Loaded {len(cookies_json)} cookies")
    print(f"📝 Important cookies found:")
    for cookie in cookies_json:
        if cookie['name'] in ['auth_token', 'ct0', 'twid']:
            print(f"   ✓ {cookie['name']}")
    
    username = os.getenv("X_USERNAME")
    password = os.getenv("X_PASSWORD")
    email = os.getenv("X_EMAIL")
    
    # Remove existing account
    try:
        await api.pool.delete_accounts(username)
        print(f"\n🗑️  Removed old account: {username}")
    except Exception as e:
        print(f"\nℹ️  No existing account to remove: {e}")
    
    # Add account with cookies
    print(f"\n➕ Adding account with cookies...")
    try:
        await api.pool.add_account(
            username=username,
            password=password,
            email=email,
            email_password=password,
            cookies=cookie_str
        )
        print(f"✅ Account added successfully!")
    except Exception as e:
        print(f"❌ Error adding account: {e}")
        return False
    
    # Check account status
    print(f"\n🔍 Verifying account status...")
    accounts = await api.pool.accounts_info()
    
    for acc in accounts:
        # Handle dict or object
        username_check = acc.get('username') if isinstance(acc, dict) else getattr(acc, 'username', None)
        
        if username_check == username:
            active = acc.get('active') if isinstance(acc, dict) else getattr(acc, 'active', False)
            email_val = acc.get('email') if isinstance(acc, dict) else getattr(acc, 'email', '')
            last_used = acc.get('last_used') if isinstance(acc, dict) else getattr(acc, 'last_used', '')
            
            print(f"\n📊 Account Details:")
            print(f"   Username: {username_check}")
            print(f"   Email: {email_val}")
            print(f"   Active: {'✅ YES' if active else '❌ NO'}")
            print(f"   Last Used: {last_used}")
            
            if active:
                print(f"\n🎉 SUCCESS! Account is active and ready to use!")
                return True
            else:
                print(f"\n⚠️  Account added but not active. May need to login again.")
                return False
    
    print(f"\n❌ Account not found after adding")
    return False

if __name__ == "__main__":
    success = asyncio.run(import_cookies())
    exit(0 if success else 1)
