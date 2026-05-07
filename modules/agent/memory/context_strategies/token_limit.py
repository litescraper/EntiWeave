import logging
from typing import Optional, Any
from transformers import AutoTokenizer

from .base import BaseContextStrategy
from modules.agent.agent_types import MessageList

class TokenLimitStrategy(BaseContextStrategy):
    """
    An 'all-or-nothing' strategy that counts tokens using a HuggingFace tokenizer.
    If the max token limit is reached, it replaces the last message with a prompt
    to force the model to provide a final answer immediately.
    """

    def __init__(self, tokenizer_path: str, max_tokens: int = 110 * 1024, max_tool_context: Optional[int] = None, reset_interval: Optional[int] = None, logger: Optional[Any] = None):
        """
        Args:
            tokenizer_path: Local path or huggingface model id for the tokenizer.
            max_tokens: The maximum number of tokens allowed in the context.
            max_tool_context: The maximum number of recent tool results to keep uncompressed.
            reset_interval: Optional number of turns after which to reset all assistant-tool messages.
            logger: Optional logger to use for tracking token counts.
        """
        super().__init__(logger=logger)
        self.tokenizer_path = tokenizer_path
        self.max_tokens = max_tokens
        self.max_tool_context = max_tool_context
        self.reset_interval = reset_interval
        self.terminal_prompt_injected = False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = None

    def _count_tokens(self, messages: MessageList) -> int:
        if not self.tokenizer:
            # Fallback heuristic if tokenizer is missing: ~4 chars per token
            return sum(len(str(m.get("content", ""))) // 4 for m in messages)
            
        try:
            # To avoid template-specific errors (like "Can only get item pairs from a mapping")
            # and to ensure consistent token counting regardless of the model's chat template 
            # support for tools/reasoning, we simplify the messages for token counting.
            # Token counting doesn't need to be 100% exact, just a good estimate.
            
            clean_messages = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content") or ""
                
                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content)
                
                # Simplify message for counting
                msg = {"role": str(role), "content": content}
                
                # If it's a tool message, some templates fail. Convert to user for counting.
                if role == "tool":
                    msg["role"] = "user"
                    msg["content"] = f"Observation: {content}"
                
                # If there's reasoning content or tool calls, include them in the content for counting
                extra_info = []
                if "reasoning_content" in m and m["reasoning_content"]:
                    extra_info.append(f"<thought>{m['reasoning_content']}</thought>")
                if "tool_calls" in m and m["tool_calls"]:
                    extra_info.append(str(m["tool_calls"]))
                
                if extra_info:
                    msg["content"] = "\n".join(extra_info) + "\n" + msg["content"]
                
                clean_messages.append(msg)

            try:
                # Try to apply chat template to accurately measure tokens with simplified messages
                full_prompt = self.tokenizer.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                # If apply_chat_template still fails, fallback to a raw string concatenation
                full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in clean_messages])

            if full_prompt is None:
                full_prompt = str(clean_messages)

            tokens = self.tokenizer(full_prompt, return_tensors="pt")
            return len(tokens["input_ids"][0])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error counting tokens: {e}")
            # Final fallback heuristic
            return sum(len(str(m.get("content", ""))) // 4 for m in messages)

    def _apply_truncation(self, messages: MessageList) -> MessageList:
        """
        Applies tool context compression if configured.
        Returns a new list of messages if compression occurs, otherwise returns the original list.
        """
        if not messages or self.max_tool_context is None:
            return messages
            
        tool_msg_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_msg_indices) <= self.max_tool_context:
            return messages
            
        num_to_compress = len(tool_msg_indices) - self.max_tool_context
        indices_to_compress = set(tool_msg_indices[:num_to_compress])
        
        new_messages = []
        for i, m in enumerate(messages):
            if i in indices_to_compress:
                new_msg = dict(m)
                new_msg["content"] = "tool_call history is compressed"
                new_messages.append(new_msg)
            else:
                new_messages.append(m)
        return new_messages

    def truncate_messages(self, messages: MessageList) -> MessageList:
        if not messages:
            return messages

        current_messages = messages
        if self.reset_interval is not None and self.reset_interval > 0:
            system_msg = None
            initial_user_msg = None
            current_idx = 0
            
            if current_idx < len(messages) and messages[current_idx].get("role") == "system":
                system_msg = messages[current_idx]
                current_idx += 1
            if current_idx < len(messages) and messages[current_idx].get("role") == "user":
                initial_user_msg = messages[current_idx]
                current_idx += 1
                
            remaining_messages = messages[current_idx:]
            assistant_indices = [i for i, msg in enumerate(remaining_messages) if msg.get("role") == "assistant"]
            num_turns = len(assistant_indices)
            
            if num_turns > 0:
                turns_to_keep = num_turns % self.reset_interval
                if turns_to_keep == 0:
                    remaining_messages = []
                else:
                    keep_from_idx = assistant_indices[-turns_to_keep]
                    remaining_messages = remaining_messages[keep_from_idx:]
                    
                current_messages = []
                if system_msg: current_messages.append(system_msg)
                if initial_user_msg: current_messages.append(initial_user_msg)
                current_messages.extend(remaining_messages)
                
                if self.logger and turns_to_keep == 0:
                    self.logger.info(f"Reset interval ({self.reset_interval}) reached. Truncating all assistant-tool messages.")

        truncated_messages = self._apply_truncation(current_messages)
        
        if self.logger and truncated_messages is not current_messages:
            num_to_compress = len([m for m in current_messages if m.get("role") == "tool"]) - self.max_tool_context
            self.logger.info(f"Compressed {num_to_compress} tool call messages")

        token_count = self._count_tokens(truncated_messages)
        if self.logger:
            rem_assistant_count = sum(1 for m in truncated_messages if m.get("role") == "assistant")
            self.logger.info(f"Current token count: {token_count} / {self.max_tokens}. Turns: {rem_assistant_count}")
        
        # TokenLimitStrategy primarily relies on max_tokens and max_tool_context
        # to manage context size, signaling a terminal prompt when exceeded.

        return truncated_messages

    def get_terminal_prompt(self, messages: MessageList) -> Optional[str]:
        """
        If the token count (after potential truncation) exceeds the limit, 
        return a prompt to force the final answer.
        Sets self.terminal_prompt_injected to True after returning the prompt.
        """
        if self.terminal_prompt_injected:
            return None
            
        # Count tokens AFTER applying truncation logic
        truncated_messages = self._apply_truncation(messages)
        token_count = self._count_tokens(truncated_messages)
        
        if token_count > self.max_tokens:
            if self.logger:
                self.logger.warning(f"Token limit exceeded ({token_count} > {self.max_tokens}). Generating terminal prompt.")
            
            self.terminal_prompt_injected = True
            return (
                "You have now reached the maximum context length you can handle. "
                "You should stop making tool calls and, based on all the information above, "
                "think again and provide what you consider the most likely answer by calling the `finish` tool."
            )
        return None

    def check_termination(self, messages: MessageList) -> bool:
        """
        Returns True if the token limit was reached and we've already 
        given the agent one last chance to provide a final answer.
        """
        # If terminal_prompt_injected is True, it means we already added the 
        # prompt in the PREVIOUS turn, so we should terminate NOW.
        return self.terminal_prompt_injected
