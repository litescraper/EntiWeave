import string
from typing import List, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
from modules.utils import http_get
from modules.types import DuckDuckGoResult


BANNED_URLS = ["https://www.wordplays.com", "https://www.storyboardthat.com", "https://etsy.com", "https://www.watchmojo.com"]

class DuckDuckGoSearcher:
    """
    DuckDuckGo Lite HTML scrape, returns structured results.
    """
    def __init__(self, max_results: int = 10):
        """
        Initialize the DuckDuckGo searcher.
        """
        self.max_results = max_results
        self.base_url = "https://lite.duckduckgo.com/lite/"

    def search(self, query: str, num: Optional[int] = None, return_debug: bool = False) -> dict:
        query = query.strip()

        # remove any punctuation in query
        query = query.translate(str.maketrans('', '', string.punctuation))
        if not query:
            return {"query": query, "results": []}
        
        limit = num if num is not None else self.max_results
        # limit the number of results to 50
        limit = max(1, min(limit, 50))

        params = {"q": query}
        html = http_get(self.base_url, params=params, referer="https://duckduckgo.com/")
        if "Unfortunately, bots use DuckDuckGo too" in html:
            raise Exception("DuckDuckGo bot detection triggered")
        
        results = self._parse_lite_results(html, max_results=limit)
        
        output = {"query": query, "results": results, "provider": "duckduckgo"}
        if return_debug:
            soup = BeautifulSoup(html, "html.parser")
            output["debug"] = {
                "result_links": len(soup.select("a.result-link")),
                "html_length": len(html),
            }
        return output

    def _parse_lite_results(self, html: str, max_results: int) -> List[DuckDuckGoResult]:
        """
        Parse the DuckDuckGo Lite HTML results.
        """
        soup = BeautifulSoup(html, "html.parser")
        results: List[DuckDuckGoResult] = []

        for link in soup.select("a.result-link"):
            title = link.get_text(" ", strip=True)
            raw_href = link.get("href", "")

            url = raw_href
            try:
                parsed = urlparse(raw_href)
                q = parse_qs(parsed.query)
                if "uddg" in q and q["uddg"]:
                    url = unquote(q["uddg"][0])
                elif raw_href.startswith("//"):
                    url = "https:" + raw_href
                
                # Filter out ads/redirect results that still point to DuckDuckGo domains
                try:
                    final_host = urlparse(url).netloc.lower()
                    if final_host.endswith("duckduckgo.com"):
                        continue
                except Exception:
                    pass
            except Exception:
                pass

            description = ""
            tr = link.find_parent("tr")
            next_tr = tr.find_next_sibling("tr") if tr else None
            seen = 0
            while next_tr and seen < 4:
                snippet_td = next_tr.find("td", class_="result-snippet")
                if snippet_td and not description:
                    description = snippet_td.get_text(" ", strip=True)
                next_tr = next_tr.find_next_sibling("tr")
                seen += 1

            if title and url:
                # filter banned url
                if any(url.startswith(banned_url) for banned_url in BANNED_URLS):
                    continue
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "description": description,
                    }
                )
                if len(results) >= max_results:
                    break

        return results

