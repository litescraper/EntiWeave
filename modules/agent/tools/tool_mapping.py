from typing import Dict, Any, Optional
from .web_tool import SearchTool, VisitPageTool, ClickTool, RetrieveTool
from .code_tool import CodeTool
from .empty_tool import EmptyTool

def get_tool_classes():
    return {
        "search": SearchTool,
        "visit": VisitPageTool,
        "open": VisitPageTool,
        "click": ClickTool,
        "retrieve": RetrieveTool,
        "python": CodeTool,
        "answer_attempt": EmptyTool
    }
