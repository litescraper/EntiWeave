import json
from typing import List, Dict, Any, Optional
from modules.agent.agent_types import Message, MessageList

class BaseMemory:
    """
    Base class for managing conversation history (memory).
    Provides basic operations for appending messages to the context.
    """
    def __init__(self, system_prompt: Optional[str] = None, messages: Optional[MessageList] = None):
        self._messages: MessageList = messages or []
        if system_prompt and not any(m.get("role") == "system" for m in self._messages):
            self.add_system_message(system_prompt)

    def add_message(self, message: Message):
        """Generic method to add a message to the history."""
        self._messages.append(message)

    def add_user_message(self, content: str):
        """Adds a user message to the history."""
        self.add_message({"role": "user", "content": content})

    def add_assistant_message(
        self, 
        content: str, 
        reasoning_content: Optional[str] = None, 
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ):
        """Adds an assistant message to the history."""
        message: Message = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.add_message(message)

    def add_tool_message(self, tool_call_id: str, content: str, tool_name: Optional[str] = None):
        """Adds a tool (observation) message to the history."""
        message: Message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        }
        if tool_name:
            message["tool_name"] = tool_name
        self.add_message(message)

    def add_system_message(self, content: str):
        """Adds a system message to the history."""
        self.add_message({"role": "system", "content": content})

    def edit_last_assistant_message(self, content: str, reasoning_content: Optional[str] = None):
        """Edits the content and reasoning_content of the last assistant message."""
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i].get("role") == "assistant":
                self._messages[i]["content"] = content
                if reasoning_content is not None:
                    self._messages[i]["reasoning_content"] = reasoning_content
                return
        raise ValueError("No assistant message found in history to edit.")

    def get_messages(self) -> MessageList:
        """Returns the full conversation history."""
        return self._messages

    def get_formatted_messages(self, sampler_type: str = "ollama", include_thinking: bool = False) -> MessageList:
        """
        Returns the conversation history formatted according to the sampler's requirements.
        
        Args:
            sampler_type: The type of sampler (e.g., "ollama", "gpt", "vllm", "claude", "glm").
            include_thinking: If True, includes reasoning_content (if present) 
                             wrapped in <think> tags or formatted appropriately.
        """
        return self.format_messages(self._messages, sampler_type=sampler_type, include_thinking=include_thinking)

    @staticmethod
    def format_messages(messages: MessageList, sampler_type: str = "ollama", include_thinking: bool = False) -> MessageList:
        """
        Formats a list of messages based on the requirements of the sampler.
        
        Args:
            messages: List of messages to format.
            sampler_type: The type of sampler (e.g., "ollama", "vllm", "gpt", "claude", "glm").
            include_thinking: If True, includes reasoning_content (if present).
        """
        formatted_messages: MessageList = []
        
        # Samplers that support native tool calls and specific message roles
        # "glm", "deepseek", "gemini" are often OpenAI-compatible
        openai_compatible_samplers = ["gpt", "vllm", "glm", "deepseek", "gemini"]
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            
            # Create the base formatted message
            formatted_msg: Dict[str, Any] = {
                "role": role,
                "content": content
            }
            
            # 1. Handle Reasoning Content (Thinking)
            reasoning = msg.get("reasoning_content")
            if reasoning and include_thinking:
                if sampler_type == "ollama":
                    # Ollama specific format: use a separate 'thinking' field
                    formatted_msg["thinking"] = reasoning
                else:
                    # For most models currently, wrapping in <think> is the most compatible way 
                    # to preserve reasoning in history, even for R1-style models on Ollama/vLLM.
                    formatted_msg["content"] = f"<think>\n{reasoning}\n</think>\n{content}".strip()
            
            # 2. Handle Tool Calls in Assistant Messages
            if role == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                if sampler_type in openai_compatible_samplers or sampler_type == "ollama":
                    # Keep as native tool calls for supported samplers
                    formatted_msg["tool_calls"] = tool_calls
                else:
                    # Fallback to XML-like format for others
                    tc_str = ""
                    for tc in tool_calls:
                        # Handle both OpenAI-style and internal-style tool calls
                        # Some samplers might return a dict with 'function': {'name': ..., 'arguments': ...}
                        if "function" in tc and isinstance(tc["function"], dict):
                            tc_name = tc["function"].get("name", "")
                            tc_args = tc["function"].get("arguments", {})
                        else:
                            tc_name = tc.get("name", "")
                            tc_args = tc.get("arguments", {})
                            
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args)
                            except:
                                pass
                                
                        tc_str += f"\n<tool_call>{tc_name}\n"
                        if isinstance(tc_args, dict):
                            for k, v in tc_args.items():
                                tc_str += f"<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>\n"
                        tc_str += "</tool_call>"
                    formatted_msg["content"] = f"{formatted_msg['content']}\n{tc_str}".strip()

            # 3. Handle Tool Response Messages
            if role == "tool":
                if sampler_type == "ollama":
                    # Ollama often expects 'tool_name' instead of 'tool_call_id'
                    tool_name = msg.get("tool_name")
                    if not tool_name and msg.get("tool_call_id"):
                        # Fallback to tool_call_id if tool_name is not stored
                        tool_name = msg["tool_call_id"]
                    
                    if tool_name:
                        formatted_msg["tool_name"] = tool_name
                elif sampler_type in openai_compatible_samplers:
                    # Keep tool_call_id for native tool support
                    if msg.get("tool_call_id"):
                        formatted_msg["tool_call_id"] = msg["tool_call_id"]
                else:
                    # For non-native, convert 'tool' role to 'user' or 'assistant' 
                    # with a prefix so the model understands it's an observation.
                    formatted_msg["role"] = "user"
                    formatted_msg["content"] = f"Observation: {content}"
            
            formatted_messages.append(formatted_msg)
            
        return formatted_messages

    def clear(self):
        """Clears the conversation history."""
        self._messages = []

