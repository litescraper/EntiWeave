from email import message
import re
import os
import json
import time
from typing import List, Dict, Any, Optional, Union

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import Message, MessageList, AgentSamplerConfig, BrowserConfig, SamplerResponse, AgentRunResult
from modules.agent.browser.simple_browser import SimpleBrowser
from modules.agent.memory.base_memory import BaseMemory

# Special tokens for Search O1
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

class SearchO1Agent(BaseLLMAgent):
    """
    A search agent implementing the Search O1 mechanism.
    It uses special tokens <|begin_search_query|> and <|end_search_query|>
    to trigger searches and processes results into the reasoning chain.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        system_prompt: Optional[str] = None,
        max_search_limit: int = 10,
    ):
        """
        Initialize the Search O1 Agent.

        Args:
            config: Configuration for the LLM sampler.
            browser_config: Configuration for the browser tool.
            system_prompt: Optional custom system prompt.
            max_search_limit: Maximum number of search attempts.
        """
        # Load prompts from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        prompt_data = self._load_prompts(yaml_path)

        # Use Search O1 system prompt if not provided
        if not system_prompt:
            system_prompt_template = prompt_data.get("search_o1_system_prompt", "")
            effective_system_prompt = system_prompt_template.format(max_search_limit=max_search_limit)
        else:
            effective_system_prompt = system_prompt
        
        # Ensure END_SEARCH_QUERY is in stop tokens for the main loop
        if config is None:
            config = AgentSamplerConfig()
        
        if config.stop is None:
            config.stop = [END_SEARCH_QUERY]
        elif isinstance(config.stop, str):
            if config.stop != END_SEARCH_QUERY:
                config.stop = [config.stop, END_SEARCH_QUERY]
            else:
                config.stop = [END_SEARCH_QUERY]
        elif isinstance(config.stop, list) and END_SEARCH_QUERY not in config.stop:
            config.stop.append(END_SEARCH_QUERY)

        super().__init__(
            config=config,
            system_prompt=effective_system_prompt,
        )
        
        self.max_search_limit = max_search_limit
        self.browser = SimpleBrowser(config=browser_config)
        self.webpage_to_reasonchain_template = prompt_data.get("webpage_to_reasonchain_prompt", "")
    
    def _extract_between(self, text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """Extracts text between two tags."""
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _extract_snippet_with_context(self, raw_context: str, snippet: str, context_chars: int = 3000) -> str:
        """Extracts a snippet with surrounding context from raw text."""
        if not snippet or snippet not in raw_context:
            return raw_context[:context_chars * 2]
            
        idx = raw_context.find(snippet)
        start = max(0, idx - context_chars)
        end = min(len(raw_context), idx + len(snippet) + context_chars)
        return raw_context[start:end]

    def _truncate_reasoning(self, history: str) -> str:
        """Truncates reasoning steps for the extraction prompt."""
        # Split by potential step markers or just by newlines
        all_lines = history.split("\n")
        formatted_steps = [line.strip() for line in all_lines if line.strip()]
        
        if len(formatted_steps) <= 10:
            return '\n\n'.join(formatted_steps)
        
        truncated = []
        for i, step in enumerate(formatted_steps):
            # Keep first few and last few lines, or lines with search tags
            if i < 3 or i >= len(formatted_steps) - 5 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                truncated.append(step)
            else:
                if not truncated or truncated[-1] != "...":
                    truncated.append("...")
        
        return '\n\n'.join(truncated)

    def _get_reasoning_history(self, memory: BaseMemory) -> str:
        """Constructs a reasoning history string from memory messages."""
        history = []
        for msg in memory.get_messages():
            if msg["role"] == "assistant":
                thought = msg.get("reasoning_content", "")
                content = msg.get("content", "")
                if thought:
                    history.append(f"<think>\n{thought}\n</think>\n{content}")
                else:
                    history.append(content)
        return "\n\n".join(history).strip()

    def _search_and_compile(self, query: str, top_k: int = 3, max_doc_len: int = 3000) -> str:
        """Searches the web and compiles information following reference logic."""
        if not query:
            return ""
            
        print(f"Executing Search O1 Query: {query}")
        search_response = self.browser.do("search", query=query)
        if search_response.status.status != "success":
            return f"Search failed: {search_response.status.message}"
            
        # Get top results
        results = search_response.data[:top_k]
        
        formatted_documents = ""
        for i, res in enumerate(results):
            open_response = self.browser.do("open", url=res.url)
            raw_context = ""
            if open_response.status.status == "success":
                raw_context = open_response.data.markdownText
            
            snippet = res.snippets[0] if res.snippets else ""
            # Clean snippet tags if any
            snippet = snippet.replace('<b>','').replace('</b>','')
            
            context = self._extract_snippet_with_context(raw_context, snippet, context_chars=max_doc_len)
            
            doc_info = {
                "title": res.title,
                "url": res.url,
                "snippet": snippet,
                "context": context
            }
            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
            
        return formatted_documents

    def _process_search_results(self, query: str, truncated_prev_reasoning: str, compiled_information: str, user_query: str) -> str:
        """Uses the LLM to process search results into reasoning steps."""
        if not compiled_information or "Search failed" in compiled_information:
            return compiled_information

        prompt = self.webpage_to_reasonchain_template.format(
            question=user_query,
            prev_reasoning=truncated_prev_reasoning,
            search_query=query,
            documents=compiled_information
        )
        
        # We call the sampler directly for this extraction step
        messages = [{"role": "user", "content": prompt}]
        
        original_stop = self.sampler.config.stop
        self.sampler.config.stop = None
        try:
            response = self.sampler(messages)
            return response.content
        finally:
            self.sampler.config.stop = original_stop

    async def run(self, user_query: str, max_steps: int = 15) -> AgentRunResult:
        """
        Runs the Search O1 loop to solve the user's query.
        """
        # Initialize memory with the system prompt
        memory = BaseMemory(system_prompt=self.system_prompt)
        memory.add_user_message(user_query)
        
        executed_queries = set()
        
        for step in range(max_steps):
            print(f"\n--- Turn {step + 1} ---")
            
            # Generate response from LLM
            start_gen = time.perf_counter()

            response = self.generate_response(
                memory.get_formatted_messages(sampler_type=self.config.sampler_type, include_thinking=True), 
                include_system=False
            )
            gen_time = time.perf_counter() - start_gen
            print(f"[Time] LLM Generation: {gen_time:.2f}s")
            
            content = response.content
            thought = response.reasoning_content
            
            # Since content doesn't include stop tokens if it stops at END_SEARCH_QUERY,
            # we check if it looks like it started a search query but didn't end it.
            if BEGIN_SEARCH_QUERY in content:
                if "<|end_search_query|" in content:
                    content = content.replace("<|end_search_query|", END_SEARCH_QUERY)
                elif END_SEARCH_QUERY not in content:
                    content += END_SEARCH_QUERY
            elif thought and BEGIN_SEARCH_QUERY in thought:
                # Handle case where search query is accidentally placed inside reasoning token (thought)
                if "<|end_search_query|" in thought:
                    thought = thought.replace("<|end_search_query|", END_SEARCH_QUERY)
                elif END_SEARCH_QUERY not in thought:
                    thought += END_SEARCH_QUERY
                
                # Extract and move to content to maintain separation of reasoning and actions
                query_in_thought = self._extract_between(thought, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                if query_in_thought:
                    # Move to content if it's not already there
                    content_query = self._extract_between(content, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                    if not content_query:
                        content = BEGIN_SEARCH_QUERY + query_in_thought + END_SEARCH_QUERY
                        # Clean the thought to keep it as pure reasoning
                        thought = thought.replace(BEGIN_SEARCH_QUERY + query_in_thought + END_SEARCH_QUERY, "").strip()

            # Parse content to get search query
            query = self._extract_between(content, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

            # Add this turn's reasoning and content to memory.
            # To allow continuing the reasoning chain in a single assistant entry,
            # we check if the last message is already an assistant message and merge.
            
            # Format the new part: reasoning + content
            # We remove the search query from the content to keep the history clean,
            # as it will be followed by the search result block.
            new_part = ""
            if thought:
                new_part += f"{thought}\n"
            
            clean_content = content
            if query:
                query_block = f"{BEGIN_SEARCH_QUERY}{query}{END_SEARCH_QUERY}"
                clean_content = content.replace(query_block, "").strip()
            
            new_part += clean_content

            last_msgs = memory.get_messages()
            if last_msgs and last_msgs[-1]["role"] == "assistant":
                merged_content = last_msgs[-1].get("content", "") + "\n" + new_part
                memory.edit_last_assistant_message(content=merged_content.strip())
            else:
                memory.add_assistant_message(content=new_part.strip())

            if query and query not in executed_queries:
                if len(executed_queries) >= self.max_search_limit:
                    print(f"Max search limit reached ({self.max_search_limit}).")
                    limit_msg = f"{BEGIN_SEARCH_RESULT}\nSearch limit reached. Please provide final answer.\n{END_SEARCH_RESULT}"
                    
                    last_msgs = memory.get_messages()
                    if last_msgs and last_msgs[-1]["role"] == "assistant":
                        new_content = last_msgs[-1].get("content", "") + "\n" + limit_msg
                        memory.edit_last_assistant_message(content=new_content)
                    else:
                        memory.add_assistant_message(content=limit_msg)
                    continue

                # Search and compile
                start_search = time.perf_counter()
                compiled_info = self._search_and_compile(query)
                search_time = time.perf_counter() - start_search
                print(f"[Time] Web Search & Fetching: {search_time:.2f}s")
                
                # Construct reasoning history for extraction prompt
                full_history = self._get_reasoning_history(memory)
                truncated_prev = self._truncate_reasoning(full_history)

                # Process search results
                start_extract = time.perf_counter()
                processed_results = self._process_search_results(query, truncated_prev, compiled_info, user_query)
                extract_time = time.perf_counter() - start_extract
                print(f"[Time] Information Extraction: {extract_time:.2f}s")
                
                # Format the result with special tokens
                result_formatted = f"{BEGIN_SEARCH_RESULT}\n{processed_results}\n{END_SEARCH_RESULT}"
                
                # Append the search result to the current assistant message.
                # This keeps the result as part of the assistant's own reasoning/action flow
                # and allows the next generation to continue from the same assistant turn.
                last_msgs = memory.get_messages()
                if last_msgs and last_msgs[-1]["role"] == "assistant":
                    new_content = last_msgs[-1].get("content", "") + "\n" + result_formatted
                    memory.edit_last_assistant_message(content=new_content)
                else:
                    memory.add_assistant_message(content=result_formatted)
                
                executed_queries.add(query)
                # Continue the loop for next generation
                continue
            
            elif query and query in executed_queries:
                repeat_msg = f"{BEGIN_SEARCH_RESULT}\nRepeated search query: {query}. Please refer to previous results.\n{END_SEARCH_RESULT}"
                
                last_msgs = memory.get_messages()
                if last_msgs and last_msgs[-1]["role"] == "assistant":
                    new_content = last_msgs[-1].get("content", "") + "\n" + repeat_msg
                    memory.edit_last_assistant_message(content=new_content)
                else:
                    memory.add_assistant_message(content=repeat_msg)
                continue
            
            if not query:
                # No search query found in the output, treat as final answer
                
                # 1. Try to extract answer wrapped in <answer> tags from current content
                answer = self._extract_between(content, "<answer>", "</answer>")
                
                if not answer:
                    memory.add_user_message("Please provide your final answer.")
                    continue
                
                return AgentRunResult(
                    answer=answer,
                    trajectory=memory.get_messages(),
                    status="success"
                )

        return AgentRunResult(
            answer="Failed to reach a final answer within maximum steps.",
            trajectory=memory.get_messages(),
            status="max_steps_reached"
        )

        
        
