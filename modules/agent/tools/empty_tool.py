from typing import Any, Dict, Optional
from .base_tool import BaseTool

class EmptyTool(BaseTool):
    @property
    def name(self) -> str:
        return "nothing"

    @property
    def description(self) -> str:
        return "An empty tool that does nothing."

    async def call(self, **kwargs: Any) -> str:
        return ""