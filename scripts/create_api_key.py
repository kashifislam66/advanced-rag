import json
import secrets
import sys

from app.api.security import hash_key

if len(sys.argv) < 3:
    sys.exit("usage: python -m scripts.create_api_key <user_id> <department> [<department> ...]\n"
             "use '*' as the department for access to everything")

user_id = sys.argv[1]
departments = [name.lower() for name in sys.argv[2:]]
api_key = secrets.token_urlsafe(32)

entry = {"key_sha256": hash_key(api_key), "user_id": user_id, "departments": departments}
print("API key (shown ONCE, give it to the user, never commit it):")
print(f"  {api_key}\n")
print("Add this entry to the JSON list in config/api_keys.json:")
print(json.dumps(entry, indent=2))