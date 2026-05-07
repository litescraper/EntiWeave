from typing import List, Dict, Any, Optional, Union
from modules.agent.samplers.gpt_sampler import GPTSampler
from modules.agent.agent_types import MessageList, SamplerConfig, SamplerResponse

class GLMSampler(GPTSampler):
    """
    Sample from Zhipu (GLM) API, built on top of GPTSampler.
    Supports GLM-specific thinking mode and reasoning content.
    """

    def _sample(self, message_list: MessageList, tools: Optional[List[Dict[str, Any]]] = None) -> SamplerResponse:
        extra_body = {}
        if self.config.think:
            extra_body = {
                "thinking": {
                    "type": "enabled",
                    "clear_thinking": False
                }
            }
        else:
            extra_body = {
                "thinking": {
                    "type": "disabled"
                }
            }

        kwargs = {
            "model": self.config.model,
            "messages": message_list,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stop": self.config.stop,
            "extra_body": extra_body
        }
        if tools:
            kwargs["tools"] = tools

        completion = self.client.chat.completions.create(**kwargs)

        message = completion.choices[0].message
        content = message.content or ""
        
        # GLM-4.0+ models often provide reasoning_content
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

