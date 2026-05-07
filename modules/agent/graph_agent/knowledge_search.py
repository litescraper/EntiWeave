import os
import json
import re
import random
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Union

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, BrowserConfig, AgentRunResult, MessageList, SamplerResponse
from modules.agent.browser.simple_browser import SimpleBrowser
from modules.agent.memory.base_memory import BaseMemory
from modules.graph.knowledge_graph.cokg import Entity
from modules.logger import BaseAgentLogger
from .utils import format_predicates

class KnowledgeSearchAgent(BaseLLMAgent):
    """
    An agent that searches for information about a specific entity and extracts predicates.
    It uses a multi-turn search-explore loop with deep page exploration.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        system_prompt: Optional[str] = None,
        max_search_results: int = 20,
        logger: Optional[BaseAgentLogger] = None,
    ):
        # Load prompts and tools from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        prompt_data = self._load_prompts(yaml_path)

        # BaseLLMAgent initialization (KnowledgeSearchAgent itself can still be used as an agent)
        super().__init__(
            config=config, 
            system_prompt="", 
            tools=prompt_data.get("knowledge_search_tools", [])
        )

        self.max_search_results = max_search_results
        self.browser = SimpleBrowser(config=browser_config)
        self.logger = logger

        # Initialize the two internal agents
        self.search_agent = BaseLLMAgent(
            config=config,
            system_prompt=prompt_data.get("knowledge_search_system_prompt", ""),
            tools=prompt_data.get("knowledge_search_tools", [])
        )
        self.search_agent.call_tool = self.call_tool

        self.extraction_agent = BaseLLMAgent(
            config=config,
            system_prompt=prompt_data.get("predicate_extraction_system_prompt", ""),
            tools=prompt_data.get("predicate_extraction_tools", [])
        )

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute search tools."""
        try:
            response = self.browser.do(tool_name, **tool_args)
            return self.browser.format_response_as_text(response)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"


    def _fetch_url_contents(self, urls: List[str]) -> tuple[List[str], List[str], List[str]]:
        """Fetch content from multiple URLs. Returns (successful_urls, all_page_content, failed_urls)."""
        all_page_content = []
        successful_urls = []
        failed_urls = []

        for url in urls:
            try:
                # Fetch page content using browser
                page_response = self.browser.do("open", url=url)
                if page_response.status.status != "success":
                    error_msg = f"Failed to fetch page {url}: {page_response.status.message}"
                    failed_urls.append(url)
                    continue

                page_content = page_response.data.markdownText
                all_page_content.append(f"{page_content}")
                successful_urls.append(url)
            except Exception as e:
                error_msg = f"Error fetching {url}: {str(e)}"
                failed_urls.append(url)

        return successful_urls, all_page_content, failed_urls

    async def _extract_predicates(
        self, 
        urls: List[str], 
        query: str, 
        entity: Entity, 
        current_predicates: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extraction Agent: Extracts or revises predicates from fetched webpage content.
        """
        # 1. Fetch content from URLs
        successful_urls, all_page_content, failed_urls = self._fetch_url_contents(urls)
        if not successful_urls:
            return f"Error: Failed to fetch content from any of the URLs: {', '.join(urls)}"

        # 2. Combine and format content
        indexed_content_parts = [f"Source [{i}]: {content}" for i, content in enumerate(all_page_content)]
        combined_page_content = "\n\n---\n\n".join(indexed_content_parts)
        
        predicates_summary = format_predicates(current_predicates, max_content_length=200)
        
        # 3. Prepare system prompt for this specific extraction run
        # Note: tools_description is NO LONGER passed here as it's handled by the sampler
        effective_extraction_system_prompt = self.extraction_agent.system_prompt.format(
            entity_name=entity.name,
            entity_category=entity.category,
            entity_description=entity.description,
            query=query,
            url=", ".join(successful_urls),
            predicates_summary=predicates_summary,
            page_content=combined_page_content[:12000] # Context limit safety
        )

        # 4. Agentic loop for extraction
        memory = BaseMemory(system_prompt=effective_extraction_system_prompt)
        memory.add_user_message("Please extract or revise predicates based on the provided webpage content.")

        response = self.extraction_agent.generate_response(
            memory.get_formatted_messages(sampler_type=self.config.sampler_type),
            include_system=False
        )
        
        tool_calls = response.tool_calls or []
        if not tool_calls:
            return None

        changes_description = []
        final_summary = None

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    pass

            if tool_name == "add":
                content_text = tool_args.get("content", "")
                topic = tool_args.get("topic", "")
                source_indices = tool_args.get("source_indices", [])
                
                relevant_urls = self._get_relevant_urls(source_indices, successful_urls)
                current_predicates.append({
                    "content": content_text,
                    "topic": topic,
                    "links": relevant_urls
                })
                changes_description.append(f"[bold green]Added[/] predicate about [bold cyan]'{topic}'[/]")

            elif tool_name == "revise":
                index = int(tool_args.get("index", -1))
                revised_content = tool_args.get("revised_content", "")
                source_indices = tool_args.get("source_indices", [])

                if 0 <= index < len(current_predicates):
                    current_predicates[index]["content"] = revised_content
                    relevant_urls = self._get_relevant_urls(source_indices, successful_urls)
                    for url in relevant_urls:
                        if url not in current_predicates[index]["links"]:
                            current_predicates[index]["links"].append(url)
                    changes_description.append(f"[bold yellow]Revised[/] predicate [[bold magenta]{index}[/]]")

            elif tool_name == "finish":
                final_summary = tool_args.get("summary")

        if final_summary:
            return final_summary
            
        return self._generate_extraction_summary(changes_description, failed_urls)

    def _get_relevant_urls(self, source_indices: Union[List[int], str], successful_urls: List[str]) -> List[str]:
        """Maps indices to actual URLs."""
        if isinstance(source_indices, str):
            try:
                source_indices = json.loads(source_indices)
            except:
                source_indices = []
        
        relevant_urls = []
        for idx in source_indices:
            try:
                idx_int = int(idx)
                if 0 <= idx_int < len(successful_urls):
                    relevant_urls.append(successful_urls[idx_int])
            except:
                continue
        return relevant_urls if relevant_urls else successful_urls

    def _generate_extraction_summary(self, changes_description: List[str], failed_urls: List[str]) -> str:
        """Generate a summary of the extraction process."""
        if changes_description:
            summary = "[bold white]Changes made:[/bold white]\n" + "\n".join(f"• {desc}" for desc in changes_description)
            if failed_urls:
                summary += f"\n\n[bold red]Failed to fetch from URLs:[/bold red] [dim]{', '.join(failed_urls)}[/dim]"
            return summary
        else:
            summary = "[dim italic]No changes were made to the predicates.[/dim italic]"
            if failed_urls:
                summary += f" [bold red]Failed to fetch from URLs:[/bold red] [dim]{', '.join(failed_urls)}[/dim]"
            return summary

    async def run(
        self, 
        entity: Entity, 
        max_steps: int = 5, 
        run_id: str = None, 
        random_k: int = 3
    ) -> AgentRunResult:
        """
        Runs the knowledge search process for a given entity.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        predicates = []

        # 1. Setup system prompt
        if entity.links:
            sampled_links = random.sample(entity.links, k=min(len(entity.links), random.choice([1, 2])))
            entity_links_str = ", ".join(sampled_links)
        else:
            entity_links_str = "None available"

        # Note: tools_description is NO LONGER passed here as it's handled by the sampler
        effective_search_system_prompt = self.search_agent.system_prompt.format(
            entity_name=entity.name,
            entity_category=entity.category,
            entity_description=entity.description,
            entity_links=entity_links_str,
            max_search_limit=max_steps
        )

        memory = BaseMemory(system_prompt=effective_search_system_prompt)
        memory.add_user_message(f"Research the entity: {entity.name}. Context: {entity.description}")

        if self.logger:
            self.logger.log_agent_action("KnowledgeSearch", f"Entity: {entity.name}", "Starting knowledge search")

        step = 0
        while step < max_steps:
            response = self.search_agent.generate_response(
                memory.get_formatted_messages(sampler_type=self.config.sampler_type, include_thinking=True),
                include_system=False
            )

            tool_calls = response.tool_calls or []
            memory.add_assistant_message(
                content=response.content,
                reasoning_content=response.reasoning_content,
                tool_calls=tool_calls
            )

            if not tool_calls:
                if self.logger:
                    self.logger.log_agent_result("KnowledgeSearch", f"No tool calls in turn {step + 1}, retrying", False)
                memory.add_user_message("Please use the `search` or `open` tool to gather more information about the entity.")
                step += 1
                continue

            all_urls = []
            extraction_queries = []
            last_search_results = None

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                tool_call_id = tool_call.get("id", f"call_{step}")

                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        pass

                if tool_name == "search":
                    query = tool_args.get("query", "")
                    if not query:
                        continue

                    if self.logger:
                        self.logger.log_agent_tool_execution("KnowledgeSearch", "search", {"query": query})
                    
                    extraction_queries.append(query)
                    search_response = self.browser.do("search", query=query, num=self.max_search_results)
                    
                    if search_response.status.status == "success":
                        search_results = search_response.data
                        if search_results:
                            last_search_results = search_results
                            num_to_select = min(random_k, len(search_results))
                            selected_results = random.sample(search_results, num_to_select)
                            all_urls.extend([res.url for res in selected_results])
                            
                            observation = f"Found {len(search_results)} results. Selected {len(selected_results)} URLs for extraction."
                        else:
                            observation = "No search results found."
                    else:
                        observation = f"Search failed: {search_response.status.message}"
                    
                    memory.add_tool_message(tool_call_id, f"Observation: {observation}", tool_name=tool_name)

                elif tool_name == "open":
                    urls = tool_args.get("urls", [])
                    if isinstance(urls, str):
                        try:
                            urls = json.loads(urls)
                        except:
                            urls = [urls]

                    for url in urls:
                        if self.logger:
                            self.logger.log_agent_tool_execution("KnowledgeSearch", "open", {"url": url})
                        all_urls.append(url)
                    
                    extraction_queries.append(f"information about {entity.name}")
                    memory.add_tool_message(tool_call_id, f"Observation: Opened {len(urls)} URLs for extraction.", tool_name=tool_name)

            all_urls = list(dict.fromkeys(all_urls))

            if all_urls:
                combined_query = extraction_queries[0] if extraction_queries else f"information about {entity.name}"
                extraction_summary = await self._extract_predicates(all_urls, combined_query, entity, predicates)

                # Retry with top-k if nothing found
                if extraction_summary is None and last_search_results:
                    num_to_select = min(random_k, len(last_search_results))
                    top_urls = [res.url for res in last_search_results[:num_to_select]]
                    if set(top_urls) != set(all_urls):
                        if self.logger:
                            self.logger.log_agent_action("KnowledgeSearch", "Retrying extraction with top search results", "No predicates found in random sample")
                        extraction_summary = await self._extract_predicates(top_urls, combined_query, entity, predicates)

                if extraction_summary is None:
                    extraction_summary = "[dim italic]No changes were made to the predicates.[/dim italic]"

                if self.logger:
                    self.logger.log_knowledge_search_turn(entity.name, step + 1, max_steps, extraction_summary)

                memory.add_user_message(f"Extraction results:\n{extraction_summary}")
            else:
                memory.add_user_message("No URLs found to extract information from.")

            step += 1

        return AgentRunResult(
            status="completed",
            answer=f"Extracted {len(predicates)} predicates.",
            trajectory=memory.get_messages(),
            metadata={"predicates": predicates}
        )
