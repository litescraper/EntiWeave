import re
import uuid
import os
import yaml
import json
import functools
import inspect
from typing import Any, Dict, List, Optional, Union, Generator

from modules.agent.agent_types import Message, MessageList, AgentSamplerConfig, SamplerResponse
from modules.agent.samplers.factory import create_sampler

class BaseLLMAgent:
    """
    Base class for LLM agents.
    
    This class handles the core logic of an LLM agent, including system instructions,
    tool definitions, and LLM interaction. It is designed to be extended by
    specific agent implementations.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the agent.

        Args:
            config: AgentSampler configuration.
            system_prompt: The system prompt defining the agent's behavior.
            tools: JSON schema of allowed tools (OpenAI tool definition format).
        """
        self.config = config or AgentSamplerConfig()
        self.sampler = create_sampler(self.config)
        
        # Load prompts and tools from agent_spec if provided in config
        spec_data = {}
        if self.config.agent_spec and os.path.exists(self.config.agent_spec):
            spec_data = self._load_prompts(self.config.agent_spec)
        
        # Priority: explicit argument > agent_spec > default (None/empty)
        self.system_prompt = system_prompt or spec_data.get("system_prompt") or spec_data.get("prompt")
        self.tools = tools or spec_data.get("tools") or []
        
        # Tracking usage
        self.runs_count = 0
        self.token_usage_history = [] # List of dicts: {"input": int, "output": int}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically wrap the 'run' method of any subclass to track usage.
        """
        super().__init_subclass__(**kwargs)
        if 'run' in cls.__dict__:
            cls.run = cls._wrap_run(cls.run)

    @staticmethod
    def _wrap_run(func):
        """
        Wrapper method to log tokens and runs for the agents.
        """
        import time
        from modules.agent.agent_types import AgentRunResult

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapped_run(self, *args, **kwargs):
                self.runs_count += 1
                start_time = time.time()
                start_in = self.sampler.total_input_tokens
                start_out = self.sampler.total_output_tokens
                try:
                    result = await func(self, *args, **kwargs)
                    
                    # Update statistics if the result is an AgentRunResult
                    if isinstance(result, AgentRunResult):
                        end_time = time.time()
                        end_in = self.sampler.total_input_tokens
                        end_out = self.sampler.total_output_tokens
                        
                        result.total_time = end_time - start_time
                        result.input_tokens = end_in - start_in
                        result.output_tokens = end_out - start_out
                        
                        # Calculate turns from trajectory (user/assistant pairs) if not already set
                        if not result.turns:
                            assistant_messages = [m for m in result.trajectory if m.get("role") == "assistant"]
                            result.turns = len(assistant_messages)
                        
                        usage = {
                            "input": result.input_tokens,
                            "output": result.output_tokens,
                            "time": result.total_time,
                            "turns": result.turns
                        }
                        self.token_usage_history.append(usage)
                        
                        if hasattr(self, "logger") and self.logger:
                            self.logger.log_agent_result(
                                self.__class__.__name__, 
                                f"Run {self.runs_count} - Time: {result.total_time:.2f}s, Turns: {result.turns}, Tokens: {result.input_tokens} in, {result.output_tokens} out", 
                                True
                            )
                    return result
                except Exception as e:
                    raise e
            return wrapped_run
        else:
            @functools.wraps(func)
            def wrapped_run(self, *args, **kwargs):
                self.runs_count += 1
                start_time = time.time()
                start_in = self.sampler.total_input_tokens
                start_out = self.sampler.total_output_tokens
                try:
                    result = func(self, *args, **kwargs)
                    
                    # Update statistics if the result is an AgentRunResult
                    if isinstance(result, AgentRunResult):
                        end_time = time.time()
                        end_in = self.sampler.total_input_tokens
                        end_out = self.sampler.total_output_tokens
                        
                        result.total_time = end_time - start_time
                        result.input_tokens = end_in - start_in
                        result.output_tokens = end_out - start_out
                        
                        # Calculate turns from trajectory if not already set
                        if not result.turns:
                            user_messages = [m for m in result.trajectory if m.get("role") == "user"]
                            result.turns = len(user_messages)
                        
                        usage = {
                            "input": result.input_tokens,
                            "output": result.output_tokens,
                            "time": result.total_time,
                            "turns": result.turns
                        }
                        self.token_usage_history.append(usage)
                        
                        if hasattr(self, "logger") and self.logger:
                            self.logger.log_agent_result(
                                self.__class__.__name__, 
                                f"Run {self.runs_count} - Time: {result.total_time:.2f}s, Turns: {result.turns}, Tokens: {result.input_tokens} in, {result.output_tokens} out", 
                                True
                            )
                    return result
                except Exception as e:
                    raise e
            return wrapped_run

    def get_total_usage(self) -> Dict[str, int]:
        """Returns the total token usage across all runs."""
        total_input = sum(u["input"] for u in self.token_usage_history)
        total_output = sum(u["output"] for u in self.token_usage_history)
        return {"input": total_input, "output": total_output, "runs": self.runs_count}

    def _load_prompts(self, yaml_path: str) -> Dict[str, Any]:
        """Loads prompt and tool configuration from a yaml file."""
        if not os.path.exists(yaml_path):
            return {}
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Interact with a specific tool. 
        
        This method should be expanded by child classes as tool availability 
        and implementation are specific to each agent's context.

        Args:
            tool_name: The name of the tool to execute.
            tool_args: The arguments for the tool.

        Returns:
            The result of the tool execution.
        """
        raise NotImplementedError("call_tool must be implemented by subclasses.")

    def generate_response(
        self,
        messages: MessageList,
        system: Optional[str] = None,
        include_system: bool = True,
    ) -> SamplerResponse:
        """
        Generate a response from the LLM based on the conversation history.

        Args:
            messages: The current conversation history.
            system: Optional override for the system prompt.
            include_system: Whether to include the system prompt in the messages.

        Returns:
            The sampler response from the LLM.
        """
        effective_system = system or self.system_prompt
        
        full_messages = list(messages)
        if include_system and effective_system:
            # Prepend system prompt if it's not already the first message
            if not messages or messages[0].get("role") != "system":
                full_messages.insert(0, {"role": "system", "content": effective_system})
    
        return self.sampler(full_messages, tools=self.tools)

    def parse_tool_calls(self, content: str) -> tuple[List[Dict[str, Any]], str]:
        """
        Parses tool calls from the LLM response content and removes them from the content.
        
        Expected format:
        <tool_call>tool_name
        <arg_key>key1</arg_key>
        <arg_value>val1</arg_value>
        ...
        </tool_call>
        
        Returns:
            A tuple of (tool_calls, cleaned_content)
            tool_calls is a list of dictionaries, each containing:
            - id: A unique identifier for the tool call.
            - name: The name of the tool.
            - arguments: A dictionary of tool arguments.
            cleaned_content is the original content with all <tool_call> blocks removed.
        """
        tool_calls = []
        
        # Pattern to find all <tool_call> blocks
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        
        # Find all matches
        matches = pattern.findall(content)
        
        for match in matches:
            lines = match.strip().split('\n')
            if not lines:
                continue
                
            tool_name = lines[0].strip()
            args = {}
            
            # Use regex to find all key-value pairs
            # We use a more robust regex that handles potential whitespace
            keys = re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>", match, re.DOTALL)
            values = re.findall(r"<arg_value>\s*(.*?)\s*</arg_value>", match, re.DOTALL)
            
            for k, v in zip(keys, values):
                args[k.strip()] = v.strip()
                
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": tool_name,
                "arguments": args
            })
            
        # Remove all <tool_call> blocks from the content
        cleaned_content = pattern.sub("", content).strip()
            
        return tool_calls, cleaned_content

    def extract_json_from_text(self, text: str) -> Optional[Any]:
        """
        Extracts JSON from text, looking for JSON code blocks (```json ... ```) or raw JSON.
        
        Args:
            text: The text to extract JSON from.
            
        Returns:
            The parsed JSON object (dict, list, etc.) or None if no valid JSON is found.
        """
        # First, try to find JSON in code blocks
        json_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        json_block_match = json_block_pattern.search(text)
        
        if json_block_match:
            json_str = json_block_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If no code block found, try to find raw JSON in the text
        # Look for the outermost JSON object/array using balanced bracket matching
        json_text = self._extract_outermost_json(text)
        if json_text:
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        return None

    def _extract_outermost_json(self, text: str) -> Optional[str]:
        """
        Extract the outermost JSON object or array from text using balanced bracket matching.

        Args:
            text: The text to search for JSON.

        Returns:
            The JSON string if found, None otherwise.
        """
        n = len(text)
        stack = []
        start = -1

        for i, char in enumerate(text):
            if char in '{[':
                if not stack:  # Found start of outermost JSON
                    start = i
                stack.append(char)
            elif char in '}]':
                if not stack:
                    continue
                if (stack[-1] == '{' and char == '}') or (stack[-1] == '[' and char == ']'):
                    stack.pop()
                    if not stack:  # Found end of outermost JSON
                        return text[start:i+1]
                else:
                    # Mismatched brackets, reset
                    stack = []
                    start = -1

        return None
