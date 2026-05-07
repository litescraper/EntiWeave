import re
import asyncio
import os
import yaml
import json
import uuid
from typing import List, Dict, Any, Optional, Union

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import Message, MessageList, AgentSamplerConfig, BrowserConfig, SamplerResponse, AgentRunResult
from modules.agent.browser.simple_browser import SimpleBrowser
from modules.agent.memory.base_memory import BaseMemory
from modules.graph.knowledge_graph.cokg import Entity
from modules.logger import BaseAgentLogger

class Fuzzifier(BaseLLMAgent):
    """
    An agent that takes an entity name and description, researches it,
    and returns a fuzzy description or interesting fact that is not unique.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[BaseAgentLogger] = None,
    ):
        """
        Initialize the Fuzzifier Agent.

        Args:
            config: Configuration for the LLM sampler.
            browser_config: Configuration for the browser tool.
            system_prompt: Optional custom system prompt.
            tools: Optional custom tool definitions.
        """
        # Load prompts and tools from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        prompt_data = self._load_prompts(yaml_path)

        # If no tools provided, use the ones from yaml (native tool call API, no tool_descriptions in prompt)
        effective_tools = tools or prompt_data.get("fuzzifier_tools", [])
        effective_system_prompt = system_prompt or prompt_data.get("fuzzifier_system_prompt", "")

        super().__init__(
            config=config,
            system_prompt=effective_system_prompt,
            tools=effective_tools
        )

        self.browser = SimpleBrowser(config=browser_config)
        self.logger = logger

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Interact with the browser tool or handle agent-specific tools.
        """
        if tool_name == "finish":
            return tool_args.get("result", "No result provided.")

        # Map 'click' tool to 'click' action in browser
        action = tool_name

        # browser.do is synchronous
        response = self.browser.do(action, **tool_args)
        formatted_response = self.browser.format_response_as_text(response)

        # Truncate web page results before passing to LLM to avoid token limit issues
        if len(formatted_response) > 1500:
            formatted_response = formatted_response[:1500] + "... [content truncated]"

        return formatted_response

    async def run(self, entity: Entity, max_steps: int = 4) -> AgentRunResult:
        """
        Researches the entity and returns a short fuzzy description (one sentence).
        """
        links_str = ""
        if entity.links:
            links_str = f"\nEntity Links: {', '.join(entity.links)}"

        user_query = (
            f"Entity Name: {entity.name}\n"
            f"Entity Description: {entity.description}{links_str}\n\n"
            "Find several key facts that describe this entity and return a SHORT description (only one sentence about the entity).\n"
            "For known entities like person/year/place, finish immediately without search.\n"
            "For internet objects, do minimal search only.\n"
            f"You have a maximum of {max_steps} steps to finish the task."
        )
        
        # Initialize memory with the system prompt
        memory = BaseMemory(system_prompt=self.system_prompt)
        memory.add_user_message(user_query)
        
        for step in range(max_steps):
            if step == max_steps - 1:
                memory.add_user_message("Observation: This is the FINAL step. You must output {'action': 'finish', ...} now. Do not search again.")

            # Generate response from LLM
            response = self.generate_response(
                memory.get_formatted_messages(sampler_type=self.config.sampler_type, include_thinking=True), 
                include_system=False
            )
            content = response.content
            
            # Use tool calls directly from response
            tool_calls = response.tool_calls or []
            
            # Log the assistant's thought/action in memory
            memory.add_assistant_message(
                content=content,
                reasoning_content=response.reasoning_content,
                tool_calls=tool_calls
            )
            
            if not tool_calls:
                # If no tool calls, it might just be a thought-only response
                continue

            # Execute tool calls (native format: tool_call["function"]["name"] / ["arguments"])
            for tool_call in tool_calls:
                tool_function = tool_call["function"]

                tool_name = tool_function["name"]
                tool_args = tool_function["arguments"]
                tool_call_id = tool_call.get("id", f"call_{step}")
                
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        pass

                if tool_name == "finish":
                    # Log the finish tool call args
                    result = tool_args.get("result", content)
                    source_indices = tool_args.get("links", [])
                    
                    # Map indices to actual URLs
                    supporting_links = []
                    
                    for idx in source_indices:
                        try:
                            idx_int = int(idx)
                            if 0 <= idx_int < len(self.browser.references):
                                supporting_links.append(self.browser.references[idx_int].url)
                        except (ValueError, TypeError):
                            continue
                    
                    self.browser.reset()

                    if self.logger:
                        self.logger.log_agent_result("Fuzzifier", f"Generated fuzzy description: {result[:100]}...", True)
                    return AgentRunResult(
                        answer=result,
                        trajectory=memory.get_messages(),
                        status="success",
                        metadata={
                            "entity_id": entity.id,
                            "entity_name": entity.name,
                            "fuzzy_description": result,
                            "links": supporting_links
                        }
                    )

                # Execute tool and get observation
                try:
                    if self.logger:
                        self.logger.log_agent_tool_execution("Fuzzifier", tool_name, tool_args)
                    observation = await self.call_tool(tool_name, tool_args)
                    memory.add_tool_message(tool_call_id, f"Observation: {observation}", tool_name=tool_name)
                except Exception as e:
                    if self.logger:
                        self.logger.log_agent_result("Fuzzifier", f"Tool execution failed: {tool_name}", False)
                    memory.add_tool_message(
                        tool_call_id,
                        f"Observation: Error executing {tool_name}: {str(e)}",
                        tool_name=tool_name
                    )
        
        return AgentRunResult(
            answer="Failed to reach a final answer within the maximum number of steps.",
            trajectory=memory.get_messages(),
            status="max_steps_reached"
        )
