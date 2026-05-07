import requests
import logging
from typing import List, Dict, Optional
from .base_browser import BaseBrowser
from ..agent_types import SearchResultItem, PageContent, BrowserConfig
from .errors import NetworkError, PageNotFoundError, InvalidActionError, InvalidSearchQueryError, ForbiddenError

logger = logging.getLogger(__name__)

class SimpleBrowser(BaseBrowser):
    def __init__(self, config: Optional[BrowserConfig] = None):
        super().__init__()
        self.config = config or BrowserConfig()
        self.base_url = self.config.base_url.rstrip('/')

        if not self.config.api_key:
            raise ValueError("API key is required")
        self.api_key = self.config.api_key
        self.headers = {}
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key

    def _fetch_search(self, query: str, num: Optional[int] = None, **kwargs) -> List[SearchResultItem]:
        """Low-level implementation for actual search request using Tools API"""
        num = num or self.config.max_results
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query, "num": num},
                headers=self.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
        except requests.HTTPError:
            if response.status_code == 400:
                raise InvalidSearchQueryError(f"Search API returned 400")
            elif response.status_code == 404:
                raise PageNotFoundError(f"Search API endpoint not found")
            raise NetworkError(f"Search API returned error {response.status_code}")
        except requests.RequestException:
            raise NetworkError(f"Search API request failed")

        data = response.json()
        
        # The tools API (tools/main.py) returns a dict with 'results' key
        # Each result item has 'title', 'url', and 'description' (from ddg.py)
        
        search_results = []
        try:
            results = data.get("results", [])
            
            for item in results:
                search_item = SearchResultItem(
                    id=0, # BaseBrowser will assign the correct ID
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippets=[item.get("description", "")]
                )
                search_results.append(search_item)
        except Exception as e:
            search_results = []
            
        return search_results

    def _fetch_page(self, url: str, **kwargs) -> PageContent:
        """Low-level implementation for actual page fetch request using Tools API"""
        try:
            response = requests.get(
                f"{self.base_url}/click",
                params={"url": url},
                headers=self.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            if response.status_code == 400:
                raise InvalidActionError(f"Page fetch API returned 400")
            elif response.status_code == 404:
                raise PageNotFoundError(f"URL not found: {url}")
            elif response.status_code == 403:
                raise ForbiddenError(f"URL is forbidden: Only use local:// URLs in local retriever mode.")
            raise NetworkError(f"Page fetch API returned error {response.status_code}")
        except requests.RequestException as e:
            raise NetworkError(f"Page fetch API request failed")

        data = response.json()
        
        # Based on tools/main.py, /click returns a dict with 'text', 'url', 'title' etc.
        page_text = data.get("text", "")

        # Truncate content if max_page_content is set in config
        if self.config.max_page_content is not None and len(page_text) > self.config.max_page_content:
            logger.info(f"Truncating page content from {len(page_text)} to {self.config.max_page_content} characters")
            page_text = page_text[:self.config.max_page_content] + "\n\n[Content truncated...]"

        # Title might not be in the response, fallback to URL or some default
        title = data.get("title") or data.get("url") or url
        
        return PageContent(
            id=0, # BaseBrowser will assign the correct ID
            title=title,
            url=url,
            markdownText=page_text
        )
