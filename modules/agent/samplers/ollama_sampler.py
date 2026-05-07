import requests
import json
import re
from typing import List, Dict, Any, Optional, Union
from modules.agent.samplers.base_sampler import SamplerBase
from modules.agent.agent_types import MessageList, SamplerConfig, SamplerResponse

class OllamaSampler(SamplerBase):
    """
    Sample from Ollama's chat API
    """

    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        if not self.config.base_url:
            self.config.base_url = "http://localhost:11434"
        self.config.base_url = self.config.base_url.rstrip('/')
        self._chat_url = f"{self.config.base_url}/api/chat"

    def _get_options(self, extra_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_ctx": self.config.num_ctx,
            "stop": self.config.stop,
            "num_predict": self.config.max_tokens, # Map max_tokens to num_predict
        }
        if extra_options:
            options.update(extra_options)
        return options

    def _sample(self, message_list: MessageList, tools: Optional[List[Dict[str, Any]]] = None) -> SamplerResponse:
        payload = {
            "model": self.config.model,
            "messages": message_list,
            "stream": False,
            "options": self._get_options()
        }
        if tools:
            payload["tools"] = tools
        
        # Only add think if it's explicitly True and supported by the backend
        # Note: Standard Ollama doesn't use this at the top level, 
        # but some custom/newer versions might.
        if self.config.think:
            payload["think"] = True

        resp = requests.post(self._chat_url, json=payload, timeout=self.config.timeout)
        if 500 <= resp.status_code < 600:
            raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
        resp.raise_for_status()
        
        data = resp.json()
        message = (data or {}).get("message") or {}
        content = message.get("content", "")
        
        # Handle reasoning/thinking content for Ollama
        reasoning = message.get("thinking")
        
        # Extract tool calls following OpenAI client syntax
        tool_calls = message.get("tool_calls")

        # Sometimes Ollama doesn't return a separate thinking field, 
        # but wraps it in <think> tags in the content
        if not reasoning and content and "<think>" in content:
            match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            elif "<think>" in content:
                # Handle cases where the closing tag might be missing
                parts = content.split("<think>", 1)
                content = parts[0].strip()
                reasoning = parts[1].strip()
                
        # Extract token usage for Ollama
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
            
        return SamplerResponse(
            content=content, 
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
