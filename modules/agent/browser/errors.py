class BrowserError(Exception):
    """Base class for browser related errors"""
    pass

class NetworkError(BrowserError):
    """Error related to network connectivity or infrastructure"""
    pass

class PageNotFoundError(BrowserError):
    """Error when a requested URL returns 404 or no content"""
    pass

class FunctionFormatError(BrowserError):
    """Error related to function call formatting"""
    pass

class InvalidActionError(BrowserError):
    """Error related to invalid browser actions (e.g. invalid link ID)"""
    pass

class InvalidSearchQueryError(BrowserError):
    """Error when a search query is invalid or empty"""
    pass

class ForbiddenError(BrowserError):
    """Error when a URL is forbidden"""
    pass

def get_error_type(e: Exception) -> str:
    """Classify the error type for better LLM feedback"""
    if isinstance(e, NetworkError):
        return "Network Error"
    elif isinstance(e, PageNotFoundError):
        return "Page Not Found"
    elif isinstance(e, FunctionFormatError):
        return "Format Error"
    elif isinstance(e, InvalidActionError):
        return "Invalid Action"
    elif isinstance(e, InvalidSearchQueryError):
        return "Invalid Search Query"
    elif isinstance(e, ForbiddenError):
        return "Forbidden Error"
    elif isinstance(e, ValueError):
        return "Value Error"
    elif isinstance(e, KeyError):
        return "Key Error"
    return "Unknown Error"
