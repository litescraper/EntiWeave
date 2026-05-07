from openai import OpenAI
import os
from typing import List, Dict, Any, Optional, Union
from modules.agent.samplers.base_sampler import SamplerBase
from modules.agent.agent_types import MessageList, SamplerConfig, SamplerResponse

class GPTSampler(SamplerBase):
    """
    Sample from OpenAI-compatible chat completion API using the OpenAI client
    """

    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        # Initialize the OpenAI client. 
        # base_url defaults to "https://api.openai.com/v1" if None.

        if self.config.api_key:
            os.environ["OPENAI_API_KEY"] = self.config.api_key

        self.client = OpenAI(
            base_url=self.config.base_url,
            # We disable internal retries as SamplerBase handles retries with backoff.
            max_retries=0, 
        )

    def _sample(self, message_list: MessageList, tools: Optional[List[Dict[str, Any]]] = None) -> SamplerResponse:
        kwargs = {
            "model": self.config.model,
            "messages": message_list,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stop": self.config.stop,
            "timeout": self.config.timeout,
        }
        if tools:
            kwargs["tools"] = tools

        completion = self.client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, 'reasoning_content', None)
        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls:
            tool_calls = [tool_call.model_dump() for tool_call in tool_calls]
        
        # Extract token usage
        usage = getattr(completion, 'usage', None)
        input_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        output_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0

        return SamplerResponse(
            content=content, 
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
