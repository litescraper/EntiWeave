from abc import ABC, abstractmethod
from typing import List, Optional, Any

from modules.agent.agent_types import MessageList

class BaseContextStrategy(ABC):
    """
    Base class for context management strategies.
    Defines how message history should be truncated or managed 
    when interacting with an LLM.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger: Optional[Any] = logger

    @abstractmethod
    def truncate_messages(self, messages: MessageList) -> MessageList:
        """
        Processes and potentially truncates a list of messages based on the strategy.

        Args:
            messages: The full list of conversation messages.

        Returns:
            A list of messages that fit within the strategy's constraints.
        """
        pass

    def get_terminal_prompt(self, messages: MessageList) -> Optional[str]:
        """
        Returns a prompt to be injected into the conversation to force 
        the agent to terminate (e.g., when a limit is approaching).
        """
        return None

    def check_termination(self, messages: MessageList) -> bool:
        """
        Optional method to determine if the context strategy dictates 
        that the agent should forcefully terminate (e.g., token limit reached).
        
        Returns True if termination is required, False otherwise.
        """
        return False
