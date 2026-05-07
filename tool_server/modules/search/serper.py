import os
import requests
import string
from typing import List, Optional, Dict, Any
from modules.types import SearchResult

class SerperSearcher:
    """
    A searcher that uses the Serper Google Search API (google.serper.dev).
    """
    def __init__(self, max_results: int = 10, timeout: int = 60):
        """
        Initialize the Serper searcher.
        """
        self.api_key = os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str, num: Optional[int] = None, return_debug: bool = False) -> dict:
        """
        Perform a search using Serper.
        """
        query = query.strip()
        if not query:
            return {"query": query, "results": [], "provider": "serper"}

        # remove any punctuation from the query
        # query = query.translate(str.maketrans('', '', string.punctuation))

        limit = num if num is not None else self.max_results
        # limit the number of results to 100 (Serper max is usually 100)
        limit = max(1, min(limit, 100))
        
        headers = {
            "X-API-KEY": self.api_key if self.api_key else "",
            "Content-Type": "application/json"
        }

        payload = {
            "q": query,
            "num": limit
        }

        try:
            # Perform the request
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()

            data = resp.json()
            results = self._parse_results(data)
            
            output = {
                "query": query,
                "results": results,
                "provider": "serper"
            }
            
            if return_debug:
                output["debug"] = {
                    "status_code": resp.status_code,
                    "target_url": resp.url,
                    "response_data": data
                }
            
            return output

        except Exception as e:
            # Re-raise to match other searchers behavior
            raise e

    def _parse_results(self, data: dict) -> List[SearchResult]:
        """
        Parse the Serper JSON results.
        """
        results: List[SearchResult] = []
        organic_results = data.get("organic", [])

        for item in organic_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": item.get("snippet", "")
            })

        return results
