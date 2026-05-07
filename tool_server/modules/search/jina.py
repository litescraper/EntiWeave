import os
import requests
from typing import List, Optional, Dict, Any
from modules.types import SearchResult
import string

class JinaSearcher:
    """
    A searcher that uses the Jina Search API (s.jina.ai).
    """
    def __init__(self, max_results: int = 10, timeout: int = 60):
        """
        Initialize the Jina searcher.
        """
        self.api_key = os.getenv("JINA_API_KEY")
        self.base_url = "https://s.jina.ai/"
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str, num: Optional[int] = None, return_debug: bool = False) -> dict:
        """
        Perform a search using Jina.
        """
        query = query.strip()
        if not query:
            return {"query": query, "results": [], "provider": "jina"}

        # remove any punctuation from the query
        # query = query.translate(str.maketrans('', '', string.punctuation))

        limit = num if num is not None else self.max_results
        # limit the number of results to 50
        limit = max(1, min(limit, 50))
        
        headers = {
            "X-Respond-With": "no-content"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        params = {
            "q": query
        }

        try:
            # Perform the request
            # We use proxies={} to explicitly bypass any global proxies, as seen in open/jina.py
            resp = requests.get(self.base_url, headers=headers, params=params, timeout=self.timeout)

            if int(resp.status_code) == 422:
                return {"query": query, "results": [], "provider": "jina"}

            resp.raise_for_status()

            results = self._parse_results(resp.text)
            
            # Limit results to requested number
            results = results[:limit]
            
            output = {
                "query": query,
                "results": results,
                "provider": "jina"
            }
            
            if return_debug:
                output["debug"] = {
                    "length": len(resp.text),
                    "status_code": resp.status_code,
                    "target_url": resp.url
                }
            
            return output

        except Exception as e:
            # Re-raise to match DuckDuckGoSearcher behavior
            raise e

    def _parse_results(self, text: str) -> List[SearchResult]:
        """
        Parse the Jina Search text results.
        Supports both same-line and multi-line formats, ignoring [index] prefixes.
        """
        results: List[SearchResult] = []
        lines = text.strip().split('\n')
        current_item: Dict[str, str] = {}

        def save_current():
            if current_item.get('title') and current_item.get('url'):
                results.append({
                    "title": current_item['title'],
                    "url": current_item['url'],
                    "description": current_item.get('description', "")
                })

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Identify the start of a new result. 
            # Usually, 'Title:' is the first field of a new result.
            if 'Title:' in line:
                # If we already have a title, it means the previous result is finished.
                if current_item.get('title'):
                    save_current()
                    current_item = {}
                
                # Extract Title (and handle potential same-line markers)
                content = line.split('Title:', 1)[1].strip()
                current_item['title'] = content
                
                # Check for other markers on the same line
                if 'URL Source:' in content:
                    parts = content.split('URL Source:', 1)
                    current_item['title'] = parts[0].strip()
                    current_item['url'] = parts[1].strip()
                    if 'Description:' in current_item['url']:
                        sub_parts = current_item['url'].split('Description:', 1)
                        current_item['url'] = sub_parts[0].strip()
                        current_item['description'] = sub_parts[1].strip()
                elif 'Description:' in content:
                    parts = content.split('Description:', 1)
                    current_item['title'] = parts[0].strip()
                    current_item['description'] = parts[1].strip()
                    
            elif 'URL Source:' in line:
                content = line.split('URL Source:', 1)[1].strip()
                current_item['url'] = content
                if 'Description:' in content:
                    parts = content.split('Description:', 1)
                    current_item['url'] = parts[0].strip()
                    current_item['description'] = parts[1].strip()
                    
            elif 'Description:' in line:
                content = line.split('Description:', 1)[1].strip()
                current_item['description'] = content

        # Save the last item
        save_current()
        return results
