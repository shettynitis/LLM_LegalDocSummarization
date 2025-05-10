import requests
from .config import API_URL

def summarize_text(text: str) -> str:
    """
    For now this is a local stub. When you're ready, set API_URL
    in config.py and this will POST to your real service.
    """
    if API_URL:
        try:
            resp = requests.post(API_URL, json={"text": text})
            resp.raise_for_status()
            return resp.json().get("summary", "")
        except Exception as e:
            # fallback to mock on error
            print(f"[summarize_text] API call failed: {e}")
            
    # ——————————————————————
    # MOCKED RESPONSE (replace me later!)
    # ——————————————————————
            
    snippet = text.strip().replace("\n", " ")
    return f"📄 MOCK SUMMARY (first 100 chars): {snippet[:100]}"