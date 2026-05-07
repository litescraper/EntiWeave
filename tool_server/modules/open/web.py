import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, Any, Optional

from modules.utils import DEFAULT_HEADERS
from modules.parse.pdf import PDFParser
from modules.parse.wiki import WikiParser
from modules.parse.custom import CustomParser
from modules.parse.json import JSONParser
from modules.parse.html import HTMLParser
from modules.types import WebOpenAPIResponse

class WebPageRetriever:
    """
    A class to retrieve and parse web pages, handling HTML, PDF, and JSON content.
    """
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = DEFAULT_HEADERS
        self.html_parser = HTMLParser()
        self.wiki_parser = WikiParser()
        self.custom_parser = CustomParser()
        self.pdf_parser = PDFParser()
        self.json_parser = JSONParser()

    def is_pdf_request(self, url: str, resp: requests.Response | None) -> bool:
        """
        Determine whether the response is a PDF based on Content-Type header
        or URL path suffix.
        """
        try:
            content_type = (resp.headers.get("Content-Type") if resp and resp.headers else "") or ""
            if "application/pdf" in content_type.lower():
                return True
        except Exception:
            pass
        try:
            parsed = urlparse(url)
            if (parsed.path or "").lower().endswith(".pdf"):
                return True
        except Exception:
            if (url or "").lower().endswith(".pdf"):
                return True
        return False

    def is_json_request(self, url: str, resp: requests.Response | None) -> bool:
        """
        Determine whether the response is JSON based on Content-Type header.
        """
        try:
            content_type = (resp.headers.get("Content-Type") if resp and resp.headers else "") or ""
            if "application/json" in content_type.lower():
                return True
        except Exception:
            pass
        return False

    def fetch(self, url: str, debug: bool = False) -> WebOpenAPIResponse:
        """
        Fetch a webpage and return its content in a structured format.
        Routes to the appropriate parser based on content type.
        """
        target = url.strip()
        if not target:
            return {"url": url, "error": "Empty URL", "provider": "local"}

        try:
            resp = requests.get(target, headers=self.headers, timeout=self.timeout)
            status_code = getattr(resp, "status_code", None)
            
            if status_code is not None and status_code >= 400:
                error_payload: WebOpenAPIResponse = {
                    "url": target,
                    "error": f"Upstream HTTP {status_code}",
                    "status": status_code,
                    "provider": "local",
                }
                if debug:
                    error_payload["debug"] = {
                        "response_headers": dict(resp.headers or {}),
                        "length_raw": len(getattr(resp, "text", "") or ""),
                    }
                return error_payload

            content_type = resp.headers.get("Content-Type", "").split(';')[0].strip().lower()

            # Routing based on content type
            if self.is_pdf_request(target, resp):
                result = self._handle_pdf(target, resp, debug)
            elif self.is_json_request(target, resp):
                result = self._handle_json(target, resp, debug)
            else:
                result = self._handle_html(target, resp, debug)
            
            result["provider"] = "local"
            return result

        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            payload: WebOpenAPIResponse = {"url": target, "error": f"Network error: {e}", "status": status, "provider": "local"}
            if debug:
                payload["debug"] = {"type": "request_exception"}
            return payload
        except Exception as e:
            payload: WebOpenAPIResponse = {"url": target, "error": f"Unexpected error: {e}", "provider": "local"}
            if debug:
                payload["debug"] = {"type": "unexpected_exception"}
            return payload

    def _handle_pdf(self, url: str, resp: requests.Response, debug: bool) -> WebOpenAPIResponse:
        pdf_bytes = getattr(resp, "content", b"") or b""
        if not pdf_bytes:
            payload: WebOpenAPIResponse = {"url": url, "error": "Empty PDF response"}
            if debug:
                payload["debug"] = {"status": resp.status_code, "length_pdf_bytes": 0}
            return payload
            
        text_content = self.pdf_parser.parse(pdf_bytes)
        if not text_content.strip():
            return {
                "url": url,
                "error": "Empty text content after PDF parsing",
                "status": resp.status_code,
                "content_type": "application/pdf"
            }
        payload: WebOpenAPIResponse = {
            "url": url,
            "text": text_content,
            "content_type": "application/pdf",
        }
        if debug:
            payload["debug"] = {
                "status": resp.status_code,
                "length_pdf_bytes": len(pdf_bytes),
                "length_text": len(text_content or ""),
            }
        return payload

    def _handle_json(self, url: str, resp: requests.Response, debug: bool) -> WebOpenAPIResponse:
        content = resp.text
        if not content.strip():
            return {
                "url": url,
                "error": "Empty text content in JSON response",
                "status": resp.status_code,
                "content_type": "application/json"
            }
        parsed_json = self.json_parser.parse(content)
        
        payload: WebOpenAPIResponse = {
            "url": url,
            "text": content,
            "content_type": "application/json",
            "json": parsed_json
        }
        if debug:
            payload["debug"] = {
                "status": resp.status_code,
                "length_raw": len(content),
                "has_json": parsed_json is not None,
            }
        return payload

    def _handle_html(self, url: str, resp: requests.Response, debug: bool) -> WebOpenAPIResponse:
        html = resp.text
        content_type = resp.headers.get("Content-Type", "text/html")
        
        host = ""
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            pass
        
        if self.wiki_parser.is_wiki_site(url):
            text_content = self.wiki_parser.parse(html, url)
        elif self.custom_parser.is_custom_site(url):
            text_content = self.custom_parser.parse(html, url)
        else:
            text_content = self.html_parser.parse(html)

        if not text_content.strip():
            return {
                "url": url,
                "error": "Empty text content after parsing",
                "status": resp.status_code,
                "content_type": content_type
            }

        # Still try to find JSON in HTML content as a fallback
        parsed_json = self.json_parser.parse(text_content) or self.json_parser.parse(html)
        
        payload: WebOpenAPIResponse = {
            "url": url, 
            "text": text_content,
            "content_type": content_type
        }
        if parsed_json is not None:
            payload["json"] = parsed_json
            
        if debug:
            payload["debug"] = {
                "status": resp.status_code,
                "length_raw": len(html),
                "length_text": len(text_content),
                "has_json": parsed_json is not None,
            }
        return payload
