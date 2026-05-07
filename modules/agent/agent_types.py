from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class ActionStatus(BaseModel):
    status: str  # "success" or "error"
    message: str
    error_type: Optional[str] = None

class SearchResultItem(BaseModel):
    id: int
    title: str
    url: str
    snippets: List[str]

class PageContent(BaseModel):
    id: int
    title: str
    url: str
    markdownText: str

class BrowserActionResponse(BaseModel):
    """Unified response for browser actions"""
    status: ActionStatus
    data: Optional[Any] = None
    action_id: str

class BrowserConfig(BaseModel):
    """
    Configuration for the browser tool.
    """
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: int = 30
    max_results: int = 10
    max_page_content: Optional[int] = None

# Sampler types
class SamplerConfig(BaseModel):
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    repetition_penalty: float = 1.1
    presence_penalty: float = 1.0
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: Optional[int] = 40
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    timeout: int = 120
    seed: Optional[int] = None
    # Ollama specific
    num_ctx: Optional[int] = 32000
    think: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class AgentSamplerConfig(BaseModel):
    """
    Configuration for an LLM agent and its underlying sampler.
    """
    model: str = "qwen3:32b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    sampler_type: str = "ollama"  # "ollama", "gpt", "vllm", or "claude"
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    presence_penalty: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    num_ctx: int = 32000
    max_retries: int = 3
    seed: Optional[int] = None
    retry_backoff_seconds: float = 1.0
    timeout: int = 120
    options: Dict[str, Any] = Field(default_factory=dict)
    think: bool = False
    stop: Optional[Union[str, List[str]]] = None
    agent_spec: Optional[str] = None  # Path to a yaml file containing prompts and tools

    def to_sampler_config(self) -> SamplerConfig:
        """
        Converts AgentSamplerConfig to the standard SamplerConfig used by samplers.
        """
        return SamplerConfig(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.options.get("max_tokens", 4096),
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            timeout=self.timeout,
            num_ctx=self.num_ctx,
            think=self.think,
            stop=self.stop,
            seed=self.seed
        )

class SamplerResponse(BaseModel):
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    input_tokens: int = 0
    output_tokens: int = 0

class AgentRunResult(BaseModel):
    """
    The result of an agent's run, containing the final answer and the message trajectory.
    """
    answer: str
    trajectory: List[Dict[str, Any]]
    status: str  # "success", "error", or "max_steps_reached"
    # Statistics
    total_time: float = 0.0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Additional data specific to the agent
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Shared chat message types
Message = Dict[str, Any]
MessageList = List[Message]
