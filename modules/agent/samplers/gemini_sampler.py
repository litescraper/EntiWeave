from modules.agent.samplers.gpt_sampler import GPTSampler
from modules.agent.agent_types import SamplerConfig

class GeminiSampler(GPTSampler):
    """
    Sample from Google's Gemini API using the OpenAI-compatible endpoint.
    Reference: https://ai.google.dev/gemini-api/docs/openai
    """

    def __init__(self, config: SamplerConfig):
        if not config.base_url:
            # Google's OpenAI-compatible endpoint
            config.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        super().__init__(config)

