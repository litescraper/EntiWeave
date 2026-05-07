from typing import TypedDict, List, Optional, Dict, Any

class SearchResult(TypedDict):
    title: str
    url: str
    description: str

class DuckDuckGoResult(SearchResult):
    pass

class SearchAPIResponse(TypedDict, total=False):
    query: str
    results: List[SearchResult]
    provider: str
    debug: Optional[Dict[str, Any]]

class WebOpenAPIResponse(TypedDict, total=False):
    url: str
    text: str
    content_type: str
    provider: str
    error: Optional[str]
    status: Optional[int]
    json: Optional[Dict[str, Any]]
    debug: Optional[Dict[str, Any]]
