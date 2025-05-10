import os

# e.g. http://fastapi_server:8000/summarize
API_URL: str = os.getenv("API_URL", "")