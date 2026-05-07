import requests
import time
import random
from typing import Optional, List, Dict

# A collection of modern User-Agents and their corresponding sec-ch-ua client hints
USER_AGENTS = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="128", "Google Chrome";v="128"',
        "sec-ch-ua-platform": '"Windows"'
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="128", "Google Chrome";v="128"',
        "sec-ch-ua-platform": '"macOS"'
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
        "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="127", "Microsoft Edge";v="127"',
        "sec-ch-ua-platform": '"Windows"'
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="128", "Google Chrome";v="128"',
        "sec-ch-ua-platform": '"Linux"'
    }
]

def get_random_headers(referer: str = "https://duckduckgo.com/") -> Dict[str, str]:
    config = random.choice(USER_AGENTS)
    
    headers = {
        "User-Agent": config["User-Agent"],
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "utf-8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "sec-ch-ua": config["sec-ch-ua"],
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": config["sec-ch-ua-platform"],
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "Connection": "keep-alive",
    }
    
    if "duckduckgo.com" in referer:
        headers["Referer"] = "https://duckduckgo.com/"
        headers["Origin"] = "https://duckduckgo.com"
    else:
        headers["Referer"] = "https://www.google.com/"
        headers["Origin"] = "https://www.google.com"
        
    return headers

DDG_HEADERS = get_random_headers("https://duckduckgo.com/")
DEFAULT_HEADERS = get_random_headers("https://www.google.com/")


def http_get(url: str, params: dict | None = None, retries: int = 3, backoff_seconds: float = 0.5, referer: str = "https://duckduckgo.com/") -> str:
  """
  Perform an HTTP GET with simple retry and exponential backoff.
  Retries on network errors, timeouts, and HTTP 5xx/429 responses.
  """
  last_error: Exception | None = None

  for attempt in range(1, max(1, retries) + 1):
    headers = get_random_headers(referer)
    try:
      resp = requests.get(url, params=params or {}, headers=headers, timeout=15)
      if "Unfortunately, bots use DuckDuckGo too" in resp.text:
        # If we hit bot detection, we should wait longer before retrying or just raise
        # For DDG lite, if we see this, retrying immediately is usually useless
        raise requests.exceptions.RequestException("Bot detection triggered", response=resp)
      resp.raise_for_status()
      return resp.text
    except requests.exceptions.RequestException as e:
      status = getattr(getattr(e, "response", None), "status_code", None)
      should_retry = (
        status is None
        or status >= 500
        or status == 429
      )
      last_error = e
      if attempt >= retries or not should_retry:
        raise
      time.sleep(backoff_seconds * (2 ** (attempt - 1)))
  if last_error:
    raise last_error
  raise RuntimeError("Unexpected error in http_get without captured exception")

def http_get_json(url: str, params: dict | None = None, retries: int = 3, backoff_seconds: float = 0.5) -> dict:
  """
  HTTP GET expecting a JSON response, with retry/backoff identical to http_get.
  """
  last_error: Exception | None = None
  for attempt in range(1, max(1, retries) + 1):
    headers = get_random_headers("https://www.google.com/")
    try:
      resp = requests.get(url, params=params or {}, headers=headers, timeout=15)
      resp.raise_for_status()
      return resp.json()
    except requests.exceptions.RequestException as e:
      status = getattr(getattr(e, "response", None), "status_code", None)
      should_retry = (
        status is None
        or status >= 500
        or status == 429
      )
      last_error = e
      if attempt >= retries or not should_retry:
        raise
      time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    except ValueError as e:
      raise e
  if last_error:
    raise last_error
  raise RuntimeError("Unexpected error in http_get_json without captured exception")

