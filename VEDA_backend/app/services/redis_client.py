import redis
import json

# Using a simple dictionary for local testing instead of fakeredis to avoid threading issues
class DictStore:
    def __init__(self):
        self.store = {}
    def set(self, k, v):
        self.store[k] = v
    def get(self, k):
        return self.store.get(k)
        
redis_instance = DictStore()

# Connect to Docker Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def save_context(session_id, paragraph_text):
    # Store text with a 10-minute expiry (TTL) so memory cleans itself
    r.setex(f"context:{session_id}", 600, paragraph_text)

def get_context(session_id):
    return r.get(f"context:{session_id}")