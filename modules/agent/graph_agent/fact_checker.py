import os
import json
import uuid
from typing import List, Dict, Any, Optional

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, BrowserConfig, AgentRunResult
from modules.agent.browser.simple_browser import SimpleBrowser
from modules.agent.memory.base_memory import BaseMemory
from modules.logger import BaseAgentLogger
from .utils import format_predicates, format_tools_description

class FactCheckerAgent(BaseLLMAgent):
    """
    An agent that validates whether a statement correctly describes an entity
    by checking a list of URLs one by one.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        logger: Optional[BaseAgentLogger] = None,
    ):
        """
        Initialize the FactChecker Agent.
        """
        # Load prompts and tools from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        self.prompt_data = self._load_prompts(yaml_path)

        # Get tools and system prompt from yaml
        self.effective_tools = self.prompt_data.get("fact_checker_tools", [])
        self.system_prompt_template = self.prompt_data.get("fact_checker_system_prompt", "")

        super().__init__(
            config=config,
            system_prompt=None, # Will be set per URL in run method
            tools=self.effective_tools
        )

        self.browser = SimpleBrowser(config=browser_config)
        self.logger = logger

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Interact with tools: skip, continue, finish.
        """
        if tool_name == "skip":
            return {
                "action": "skip",
                "reasoning": tool_args.get("reasoning", "")
            }
        elif tool_name == "continue":
            return {
                "action": "continue",
                "reasoning": tool_args.get("reasoning", ""),
                "remaining_statement": tool_args.get("remaining_statement", "")
            }
        elif tool_name == "finish":
            return {
                "action": "finish",
                "reasoning": tool_args.get("reasoning", "")
            }
        
        raise NotImplementedError(f"Tool {tool_name} not implemented in FactCheckerAgent")

    async def run(
        self, 
        entity_name: str, 
        statement: str, 
        urls: List[str],
        max_steps_per_url: int = 2
    ) -> AgentRunResult:
        """
        Validates the statement against the entity name by checking URLs one by one.
        Supports partial verification across multiple URLs.
        
        Returns:
            AgentRunResult: answer will be "True" or "False".
        """
        all_trajectories = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        current_statement = statement
        evidence_list = [] # List of {reasoning, url}
        is_fully_supported = False

        for url in urls:
            if self.logger:
                self.logger.log_agent_action("FactChecker", f"Checking URL: {url}", f"Entity: {entity_name}, Remaining Statement: {current_statement}")

            # 1. Fetch content of the URL
            try:
                page_response = self.browser.do("open", url=url)
                if page_response.status.status != "success":
                    if self.logger:
                        self.logger.log_agent_result("FactChecker", f"Failed to open URL: {url}", False)
                    continue
                
                page_content = page_response.data.markdownText
                # Truncate content to avoid token limits    
                if len(page_content) > 100000:
                    page_content = page_content[:100000] + "\n... [content truncated]"
            except Exception as e:
                if self.logger:
                    self.logger.log_agent_result("FactChecker", f"Error opening URL {url}: {str(e)}", False)
                continue

            # 2. Set up the system prompt for this specific URL and current_statement
            tools_description = format_tools_description(self.effective_tools)
            system_prompt = self.system_prompt_template.format(
                entity_name=entity_name,
                statement=current_statement,
                tools_description=tools_description
            )
            
            # 3. Start ReAct loop for this URL
            memory = BaseMemory(system_prompt=system_prompt)
            memory.add_user_message(f"Webpage Content from {url}:\n\n{page_content}")
            
            for step in range(max_steps_per_url):
                response = self.generate_response(
                    memory.get_formatted_messages(sampler_type=self.config.sampler_type, include_thinking=True),
                    include_system=False
                )

                
                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens
                
                content = response.content

                # Use tool calls directly from response
                tool_calls = response.tool_calls or []
                
                memory.add_assistant_message(
                    content=content,
                    reasoning_content=response.reasoning_content,
                    tool_calls=tool_calls
                )
                
                if not tool_calls:
                    memory.add_user_message("Observation: No tool call found. Please use `skip`, `continue`, or `finish`.")
                    continue
                
                # Handle tool calls
                found_action = False
                for tool_call in tool_calls:
                    action_result = await self.call_tool(tool_call["function"]["name"], tool_call["function"]["arguments"])
                    action = action_result["action"]
                    
                    if action == "skip":
                        if self.logger:
                            self.logger.log_agent_result("FactChecker", f"Skipped {url}: {action_result['reasoning']}", False)
                        found_action = True
                        break
                    elif action == "continue":
                        reasoning = action_result["reasoning"]
                        current_statement = action_result["remaining_statement"]
                        evidence_list.append({"reasoning": reasoning, "url": url})
                        if self.logger:
                            self.logger.log_agent_result("FactChecker", f"Partial support from {url}: {reasoning}. Remaining: {current_statement}", True)
                        found_action = True
                        break
                    elif action == "finish":
                        reasoning = action_result["reasoning"]
                        evidence_list.append({"reasoning": reasoning, "url": url})
                        is_fully_supported = True
                        if self.logger:
                            self.logger.log_agent_result("FactChecker", f"Full support achieved at {url}: {reasoning}", True)
                        found_action = True
                        break
                
                if found_action:
                    break
            
            all_trajectories.extend(memory.get_messages())
            
            if is_fully_supported:
                return AgentRunResult(
                    answer="True",
                    trajectory=all_trajectories,
                    status="success",
                    metadata={
                        "entity_name": entity_name,
                        "statement": statement,
                        "evidence": evidence_list,
                        "reasoning": "; ".join([e["reasoning"] for e in evidence_list])
                    }
                )

        # If we went through all URLs and never reached 'finish'
        return AgentRunResult(
            answer="False",
            trajectory=all_trajectories,
            status="success",
            metadata={
                "entity_name": entity_name,
                "statement": statement,
                "evidence": evidence_list,
                "reasoning": "Statement not fully supported by the provided URLs. Partial evidence: " + "; ".join([e["reasoning"] for e in evidence_list]) if evidence_list else "No supporting evidence found."
            }
        )
