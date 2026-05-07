import time
from typing import List, Dict, Any, Optional
from modules.agent.agent_types import MessageList, SamplerConfig, SamplerResponse

class SamplerBase:
    """
    Base class for defining a sampling model with built-in retry logic.
    """

    def __init__(
        self,
        config: SamplerConfig,
    ):
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def __call__(self, message_list: MessageList, tools: Optional[List[Dict[str, Any]]] = None) -> SamplerResponse:
        """
        Main entry point for sampling with exponential backoff retry logic.
        """
        last_exc = None
        max_retries = self.config.max_retries
        for attempt in range(max(1, max_retries)):
            try:
                response = self._sample(message_list, tools=tools)
                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens
                return response
            except Exception as e:
                last_exc = e
                # Don't sleep if it's the last attempt
                if attempt >= max_retries - 1:
                    break
                
                delay = float(self.config.retry_backoff_seconds) * (2 ** attempt)
                # You might want to add a small jitter here, but keeping it simple for now
                time.sleep(max(0.0, delay))
        
        if last_exc:
            raise last_exc
        return SamplerResponse(content="")

    def _sample(self, message_list: MessageList, tools: Optional[List[Dict[str, Any]]] = None) -> SamplerResponse:
        """
        Internal method to be implemented by subclasses for the actual API call.
        Should perform exactly one attempt.
        """
        raise NotImplementedError
