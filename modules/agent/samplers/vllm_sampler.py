import requests
import json
from typing import List, Dict, Any, Optional, Union
from modules.agent.samplers.base_sampler import SamplerBase
from modules.agent.agent_types import MessageList, SamplerConfig, SamplerResponse

class VLLMSampler(SamplerBase):
    """
    Sample from vLLM's OpenAI-compatible API
    """

    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        if not self.config.base_url:
            self.config.base_url = "http://localhost:8000/v1"
        self.config.base_url = self.config.base_url.rstrip('/')

    def _sample(self, message_list: MessageList, tools: Optional[List[Dict[str, Any]]] = None) -> SamplerResponse:
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "Bearer EMPTY"
        }

        payload = {
            "model": self.config.model,
            "messages": message_list,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "repetition_penalty": self.config.repetition_penalty,
            "presence_penalty": self.config.presence_penalty,
            "top_p": self.config.top_p,
            "stop": self.config.stop,
        }
        if tools:
            payload["tools"] = tools
        if self.config.seed:
            payload["seed"] = self.config.seed

        resp = requests.post(url, json=payload, headers=headers, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        
        message = data["choices"][0]["message"]
        content = message.get("content", "")

        if message.get("reasoning"):
            reasoning = message.get("reasoning")
        else:
            reasoning = message.get("reasoning_content")

        tool_calls = message.get("tool_calls")
        # Extract token usage
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)


        if content == None:
            content = ""
        if reasoning == None:
            reasoning = ""
        if tool_calls == None:
            tool_calls = []

        return SamplerResponse(
            content=content, 
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
