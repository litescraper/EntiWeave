from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import uuid
import logging
import hashlib
from ..agent_types import SearchResultItem, PageContent, BrowserActionResponse, ActionStatus
from .errors import get_error_type, BrowserError

logger = logging.getLogger(__name__)

class BaseBrowser(ABC):
    def __init__(self, truncate_context: bool = True, max_context_length: int = 30000):
        self.session_id = str(uuid.uuid4())
        
        # 1. histories of retrieved references (search results only)
        self.references: List[SearchResultItem] = []
        
        # 2. histories of search queries
        self.historical_search_queries: List[str] = []
        
        # 3. dictionary (map) of pages and retrieved page content, keyed by URL hash
        self.pages: Dict[str, PageContent] = {}
        
        # Browser state tracking
        self.current_page_hash: Optional[str] = None
        self.page_history: List[str] = [] # List of URL hashes
        self.action_history: List[Any] = []

        self.truncate_context: bool = truncate_context
        self.max_context_length: int = max_context_length

    def _get_url_hash(self, url: str) -> str:
        """Generate a hash for a given URL to use as a cache key"""
        return hashlib.md5(url.encode()).hexdigest()

    def search(self, query: str, **kwargs) -> BrowserActionResponse:
        """General processing for a search action"""
        action_id = str(uuid.uuid4())
        
        # Que
        if not query or not query.strip():
            return BrowserActionResponse(
                status=ActionStatus(
                    status="error", 
                    message="Search query cannot be empty",
                    error_type="Invalid Search Query"
                ),
                action_id=action_id
            )

        self.historical_search_queries.append(query)
        
        try:
            # Call low-level implementation to fetch results
            results = self._fetch_search(query, **kwargs)
            
            # Process and assign unique IDs within the browser session
            start_id = len(self.references)
            for i, result in enumerate(results):
                result.id = start_id + i
            
            self.references.extend(results)
            
            return BrowserActionResponse(
                status=ActionStatus(status="success", message=f"Search successful, found {len(results)} results"),
                data=results,
                action_id=action_id
            )
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            error_type = get_error_type(e)
            return BrowserActionResponse(
                status=ActionStatus(
                    status="error", 
                    message=str(e),
                    error_type=error_type
                ),
                action_id=action_id
            )

    def open(self, url: str, **kwargs) -> BrowserActionResponse:
        """General processing for opening a URL, including caching"""
        action_id = str(uuid.uuid4())
        
        if not url or not url.strip():
            return BrowserActionResponse(
                status=ActionStatus(
                    status="error", 
                    message="URL cannot be empty",
                    error_type="Invalid Action"
                ),
                action_id=action_id
            )

        url_hash = self._get_url_hash(url)
        
        # Check cache first
        if url_hash in self.pages:
            logger.info(f"Retrieving {url} from cache (hash: {url_hash})")
            if self.current_page_hash and self.current_page_hash != url_hash:
                self.page_history.append(self.current_page_hash)
            self.current_page_hash = url_hash
            return BrowserActionResponse(
                status=ActionStatus(status="success", message="Page retrieved from cache"),
                data=self.pages[url_hash],
                action_id=action_id
            )
            
        try:
            # Call low-level implementation to fetch page content
            page_content = self._fetch_page(url, **kwargs)
            
            # Cache the page content
            page_content.id = len(self.pages) # Internal ID for PageContent
            self.pages[url_hash] = page_content
            
            if self.current_page_hash:
                self.page_history.append(self.current_page_hash)
            self.current_page_hash = url_hash
            
            return BrowserActionResponse(
                status=ActionStatus(status="success", message="Page opened successfully"),
                data=page_content,
                action_id=action_id
            )
        except Exception as e:
            logger.error(f"Failed to open URL '{url}': {str(e)}")
            error_type = get_error_type(e)
            return BrowserActionResponse(
                status=ActionStatus(
                    status="error", 
                    message=str(e),
                    error_type=error_type
                ),
                action_id=action_id
            )

    def click_link(self, link_id: int, **kwargs) -> BrowserActionResponse:
        """General processing for clicking a link from search results"""
        action_id = str(uuid.uuid4())
        try:
            link_id = int(link_id)
            if 0 <= link_id < len(self.references):
                url = self.references[link_id].url
                # Reuse open logic which handles caching and state
                # remove url from kwargs if any
                if "url" in kwargs:
                    del kwargs["url"]
                response = self.open(url, **kwargs)
                response.action_id = action_id # Maintain original action_id if desired or update it
                return response
            else:
                return BrowserActionResponse(
                    status=ActionStatus(
                        status="error", 
                        message=f"Invalid link ID: {link_id}. Range: 0-{len(self.references)-1}",
                        error_type="Invalid Action"
                    ),
                    action_id=action_id
                )
        except (ValueError, TypeError) as e:
            return BrowserActionResponse(
                status=ActionStatus(
                    status="error", 
                    message=f"Link ID must be an integer, got {link_id}, error: {str(e)}",
                    error_type="Invalid Action"
                ),
                action_id=action_id
            )
        except Exception as e:
            logger.error(f"Click link failed for ID {link_id}: {str(e)}")
            error_type = get_error_type(e)
            return BrowserActionResponse(
                status=ActionStatus(
                    status="error", 
                    message=str(e),
                    error_type=error_type
                ),
                action_id=action_id
            )

    @abstractmethod
    def _fetch_search(self, query: str, **kwargs) -> List[SearchResultItem]:
        """Low-level implementation for actual search request"""
        pass

    @abstractmethod
    def _fetch_page(self, url: str, **kwargs) -> PageContent:
        """Low-level implementation for actual page fetch request"""
        pass

    def do(self, action: str, **params) -> BrowserActionResponse:
        """Route the action to the corresponding general handler"""
        action_id = str(uuid.uuid4())

        try:
            if action == 'search':
                response = self.search(**params)
            elif action in ['open', 'visit']:
                response = self.open(**params)
            elif action == 'click':
                response = self.click_link(**params)
            else:
                response = BrowserActionResponse(
                    status=ActionStatus(status="error", message=f"Invalid action: {action}"),
                    action_id=action_id
                )
        except Exception as e:
            logger.error(f"Action '{action}' failed: {str(e)}")
            error_type = get_error_type(e)
            response = BrowserActionResponse(
                status=ActionStatus(
                    status="error", 
                    message=str(e),
                    error_type=error_type
                ),
                action_id=action_id
            )

        # Update action history
        self.action_history.append({
            "action_id": response.action_id,
            "action": action,
            "params": params,
            "status": response.status.status,
            "message": response.status.message
        })

        return response

    def format_response_as_text(self, response: BrowserActionResponse) -> str:
        """Format the browser action response in a way that is easily readable by an LLM."""
        status_line = f"Status: {response.status.status.upper()}"
        message_line = f"Message: {response.status.message}"
        
        output = [status_line, message_line]
        
        if response.data:
            if isinstance(response.data, list) and len(response.data) > 0 and isinstance(response.data[0], SearchResultItem):
                # Format search results
                # output.append("\n--- Search Results ---")
                output.append(self._format_search_results_as_text(response.data))
            elif isinstance(response.data, PageContent):
                # Format page content
                # output.append("\n--- Page Content ---")
                output.append(f"Title: {response.data.title}")
                output.append(f"URL: {response.data.url}")
                output.append("\nContent:")
                # Truncate content if too long for the LLM to process in one go
                content = response.data.markdownText
                if self.truncate_context and len(content) > self.max_context_length:
                    content = content[:self.max_context_length] + "\n... (content truncated) ..."
                output.append(content)
            else:
                output.append(f"\nData: {str(response.data)}")
        
        return "\n".join(output)

    def _format_search_results_as_text(self, search_results: List[SearchResultItem]) -> str:
        """Format the search results as text for observation if needed"""
        formatted_results = []
        for result in search_results:
            snippets_text = ' ... '.join([snippet.strip() for snippet in result.snippets])
            result_str = f"【{result.id}†{result.title}†{result.url}】\n{snippets_text}\n"
            formatted_results.append(result_str)
        return "\n".join(formatted_results)

    def reset(self):
        """Reset the browser state"""
        self.references = []
        self.historical_search_queries = []
        self.pages = {}
        self.current_page_hash = None
        self.page_history = []
        self.action_history = []
