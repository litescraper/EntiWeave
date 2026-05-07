from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """
    Abstract base class for all tools used by the agent.
    """
    
    def __init__(self):
        self._metadata: Dict[str, Any] = {}

    @property
    def metadata(self) -> Dict[str, Any]:
        """Tool-specific metadata (e.g., internal LLM trajectories)."""
        if not hasattr(self, "_metadata"):
            self._metadata = {}
        return self._metadata

    def reset_metadata(self) -> None:
        """Resets the tool's metadata."""
        self._metadata = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A description of what the tool does."""
        pass

    @abstractmethod
    async def call(self, **kwargs: Any) -> str:
        """
        Executes the tool with the given arguments.
        Arguments may include shared resources (like 'browser' or 'interpreter').
        Returns the observation as a string.
        """
        pass

    def format_response(self, response: Any) -> str:
        """
        Formats the tool's raw output into a string for the agent.
        """
        return str(response)
