import anthropic
import os
from typing import List, Dict, Any, Optional, Union
from modules.agent.samplers.base_sampler import SamplerBase
from modules.agent.agent_types import MessageList, SamplerConfig, SamplerResponse

class ClaudeSampler(SamplerBase):
    """
    Sample from Anthropic's Claude API using the official anthropic client.
    """

    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        # Initialize the Anthropic client. 
        # base_url is optional and can be used for proxies or Bedrock/Vertex if configured.
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=self.config.base_url,
        )

    def _sample(self, message_list: MessageList, tools: Optional[List[Dict[str, Any]]] = None) -> SamplerResponse:
        system_prompt = ""
        filtered_messages = []
        for msg in message_list:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Anthropic's messages.create handles the call.
        # Note: Claude 3.7+ supports thinking mode if enabled via the config.
        
        extra_kwargs = {}
        if self.config.think:
            # Thinking mode for Claude 3.7+
            # We use a heuristic for budget tokens if not specified
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(self.config.max_tokens // 2, 16000) 
            }
            # When thinking is enabled, temperature must be 1.0
            extra_kwargs["temperature"] = 1.0
        else:
            extra_kwargs["temperature"] = self.config.temperature

        if self.config.stop:
            if isinstance(self.config.stop, str):
                extra_kwargs["stop_sequences"] = [self.config.stop]
            else:
                extra_kwargs["stop_sequences"] = self.config.stop

        if tools:
            # Anthropic tool format is slightly different, but let's assume 
            # the caller provides the correct format for now or we might need 
            # to convert it. Standard OpenAI tool format usually works with 
            # some minor adjustments.
            extra_kwargs["tools"] = tools

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            system=system_prompt.strip() or anthropic.NOT_GIVEN,
            messages=filtered_messages,
            **extra_kwargs
        )
        
        content = ""
        reasoning = None
        tool_calls = None # TODO: Implement tool call extraction for Claude if needed
        
        # Anthropic response.content is a list of blocks
        for block in response.content:
            if block.type == 'text':
                content += block.text
            elif block.type == 'thinking':
                reasoning = (reasoning or "") + block.thinking
        
        # Extract token usage for Claude
        input_tokens = getattr(response.usage, 'input_tokens', 0)
        output_tokens = getattr(response.usage, 'output_tokens', 0)
        
        return SamplerResponse(
            content=content, 
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

