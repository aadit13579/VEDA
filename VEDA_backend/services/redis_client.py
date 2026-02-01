import redis
import json

# Connect to Docker Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def save_context(session_id, paragraph_text):
    # Store text with a 10-minute expiry (TTL) so memory cleans itself
    r.setex(f"context:{session_id}", 600, paragraph_text)

def get_context(session_id):
    return r.get(f"context:{session_id}")