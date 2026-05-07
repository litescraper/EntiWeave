import os
import requests
from typing import Optional, Dict, Any
from modules.types import WebOpenAPIResponse

class JinaWebReader:
    """
    A reader that uses the Jina Reader API (r.jina.ai) to fetch and convert web pages to Markdown.
    """
    def __init__(self, timeout: int = 60):
        self.api_key = os.getenv("JINA_API_KEY")
        self.base_url = "https://r.jina.ai/"
        self.timeout = timeout

    def read(self, url: str, debug: bool = False) -> WebOpenAPIResponse:
        target = url.strip()
        if not target:
            return {"url": url, "error": "Empty URL", "provider": "jina"}

        headers = {
            "Accept": "text/plain", # Request Markdown format
            "X-Return-Format": "markdown",
            "X-Md-Link-Style": "discarded",
            "X-Retain-Images": "none",
            "X-With-Links-Summary": "true"
        }

        if self.base_url == "https://r.jina.ai/" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # Jina Reader API works by prepending r.jina.ai/ to the URL
            target_url = f"{self.base_url}{target}"
            
            # Using a longer timeout for Jina as it might do complex rendering
            # Explicitly do not use any proxy
            resp = requests.get(target_url, headers=headers, timeout=self.timeout, proxies={})
            status_code = getattr(resp, "status_code", None)

            if status_code is not None and status_code >= 400:
                error_payload: WebOpenAPIResponse = {
                    "url": target,
                    "error": f"Jina Upstream HTTP {status_code}",
                    "status": status_code,
                    "provider": "jina",
                }
                if debug:
                    error_payload["debug"] = {
                        "response_headers": dict(resp.headers or {}),
                        "target_url": target_url
                    }
                return error_payload

            text = resp.text
            if not text.strip():
                return {
                    "url": target,
                    "error": "Empty content from Jina",
                    "status": status_code,
                    "provider": "jina"
                }

            result: WebOpenAPIResponse = {
                "url": target,
                "text": text,
                "content_type": "text/markdown",
                "provider": "jina",
                "status": status_code
            }
            if debug:
                result["debug"] = {"length": len(text)}
            return result

        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            return {
                "url": target, 
                "error": f"Jina network error: {e}", 
                "status": status, 
                "provider": "jina"
            }
        except Exception as e:
            return {
                "url": target, 
                "error": f"Jina unexpected error: {e}", 
                "provider": "jina"
            }
