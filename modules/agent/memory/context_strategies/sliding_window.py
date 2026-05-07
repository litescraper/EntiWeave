from typing import Optional, Any
from .base import BaseContextStrategy
from modules.agent.agent_types import MessageList

class SlidingWindowStrategy(BaseContextStrategy):
    """
    A strategy that keeps only the last X conversational turns,
    while always retaining the system prompt and the initial user prompt.
    """

    def __init__(self, max_conversational_turns: Optional[int] = None, 
                 tokenizer: Optional[Any] = None, max_tokens: Optional[int] = None,
                 logger: Optional[Any] = None):
        """
        Args:
            max_conversational_turns: Optional limit on the number of 
                                      conversational turns to keep in context.
            tokenizer: Optional tokenizer to count tokens.
            max_tokens: Optional max token limit. If context exceeds this, older turns are removed.
            logger: Optional logger to use for tracking truncation.
        """
        super().__init__(logger=logger)
        self.max_conversational_turns = max_conversational_turns
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def _count_tokens(self, messages: MessageList) -> int:
        if not messages:
            return 0
        
        if not self.tokenizer:
            # Fallback heuristic if tokenizer is missing: ~4 chars per token
            return sum(len(str(m.get("content", ""))) // 4 for m in messages)
            
        try:
            # Simplify messages for consistent token counting across different templates
            clean_messages = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                
                msg = {"role": str(role), "content": content}
                if role == "tool":
                    msg["role"] = "user"
                    msg["content"] = f"Observation: {content}"
                
                extra_info = []
                if "reasoning_content" in m and m["reasoning_content"]:
                    extra_info.append(f"<thought>{m['reasoning_content']}</thought>")
                if "tool_calls" in m and m["tool_calls"]:
                    extra_info.append(str(m["tool_calls"]))
                
                if extra_info:
                    msg["content"] = "\n".join(extra_info) + "\n" + msg["content"]
                clean_messages.append(msg)

            # Try to apply chat template to accurately measure tokens
            full_prompt = getattr(self.tokenizer, "apply_chat_template", None)
            if full_prompt is not None:
                try:
                    prompt_str = self.tokenizer.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=False)
                except Exception:
                    prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in clean_messages])
                if prompt_str is None:
                    prompt_str = str(clean_messages)
            else:
                prompt_str = str(clean_messages)
            
            # Use tokenizer encoding to count
            if callable(self.tokenizer):
                tokens = self.tokenizer(prompt_str, return_tensors="pt")
                return len(tokens["input_ids"][0])
            elif hasattr(self.tokenizer, "encode"):
                return len(self.tokenizer.encode(prompt_str))
            else:
                # Fallback if tokenizer object is weird
                return sum(len(str(m.get("content", ""))) // 4 for m in messages)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error counting tokens: {e}")
            return sum(len(str(m.get("content", ""))) // 4 for m in messages)

    def truncate_messages(self, messages: MessageList) -> MessageList:
        if not messages:
            return messages

        # 1. Identify System and Initial User prompt
        system_msg = None
        initial_user_msg = None
        
        current_idx = 0
        if current_idx < len(messages) and messages[current_idx].get("role") == "system":
            system_msg = messages[current_idx]
            current_idx += 1
            
        if current_idx < len(messages) and messages[current_idx].get("role") == "user":
            initial_user_msg = messages[current_idx]
            current_idx += 1
            
        # The rest of the messages are the "conversational" part we might truncate
        remaining_messages = messages[current_idx:]
        if not remaining_messages:
            return messages

        # 2. Find indices of all assistant messages in the remaining part
        assistant_indices = [i for i, msg in enumerate(remaining_messages) if msg.get("role") == "assistant"]
        
        # Determine how many turns we currently have
        num_turns = len(assistant_indices)
        
        # 3. Truncate by max_conversational_turns if applicable
        if self.max_conversational_turns is not None and self.max_conversational_turns > 0:
            if num_turns > self.max_conversational_turns:
                if self.logger:
                    self.logger.info(f"Truncating context: {num_turns} turns -> {self.max_conversational_turns} turns.")
                num_turns = self.max_conversational_turns
                keep_from_idx = assistant_indices[-num_turns]
                remaining_messages = remaining_messages[keep_from_idx:]
                assistant_indices = [i for i, msg in enumerate(remaining_messages) if msg.get("role") == "assistant"]
        
        # 4. Reconstruct the message list
        def build_messages(rem_msgs):
            new_messages = []
            if system_msg:
                new_messages.append(system_msg)
            if initial_user_msg:
                new_messages.append(initial_user_msg)
            new_messages.extend(rem_msgs)
            return new_messages

        current_messages = build_messages(remaining_messages)

        # 5. Truncate by max_tokens if applicable
        if self.max_tokens is not None and self.max_tokens > 0:
            while num_turns > 0:
                token_count = self._count_tokens(current_messages)
                if token_count <= self.max_tokens:
                    break
                
                # Exceeds max_tokens, remove the oldest turn
                if self.logger:
                    self.logger.info(f"Token count {token_count} exceeds max {self.max_tokens}. Removing a turn.")
                
                num_turns -= 1
                if num_turns == 0:
                    # Removed all conversational turns
                    remaining_messages = []
                    current_messages = build_messages(remaining_messages)
                    break
                
                keep_from_idx = assistant_indices[-num_turns]
                remaining_messages = remaining_messages[keep_from_idx:]
                # Re-evaluate assistant indices relative to new remaining_messages
                assistant_indices = [i for i, msg in enumerate(remaining_messages) if msg.get("role") == "assistant"]
                current_messages = build_messages(remaining_messages)

        return current_messages
