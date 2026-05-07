from typing import Optional
from modules.agent.agent_types import AgentSamplerConfig
from modules.agent.samplers.base_sampler import SamplerBase
from modules.agent.samplers.ollama_sampler import OllamaSampler
from modules.agent.samplers.gpt_sampler import GPTSampler
from modules.agent.samplers.vllm_sampler import VLLMSampler
from modules.agent.samplers.claude_sampler import ClaudeSampler
from modules.agent.samplers.glm_sampler import GLMSampler
from modules.agent.samplers.gemini_sampler import GeminiSampler
from modules.agent.samplers.deepseek_sampler import DeepSeekSampler


def create_sampler(config: AgentSamplerConfig) -> SamplerBase:
    """
    Factory function to create a sampler based on the provided configuration.
    """
    sampler_cfg = config.to_sampler_config()
    
    if config.sampler_type == "ollama":
        return OllamaSampler(sampler_cfg)
    elif config.sampler_type == "gpt":
        return GPTSampler(sampler_cfg)
    elif config.sampler_type == "vllm":
        return VLLMSampler(sampler_cfg)
    elif config.sampler_type == "claude":
        return ClaudeSampler(sampler_cfg)
    elif config.sampler_type == "glm":
        return GLMSampler(sampler_cfg)
    elif config.sampler_type == "gemini":
        return GeminiSampler(sampler_cfg)
    elif config.sampler_type == "deepseek":
        return DeepSeekSampler(sampler_cfg)
    else:
        raise ValueError(f"Unknown sampler type: {config.sampler_type}")

