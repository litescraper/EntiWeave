import re
import asyncio
import os
import yaml
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import Message, MessageList, AgentSamplerConfig, BrowserConfig, SamplerResponse, AgentRunResult
from modules.agent.browser.simple_browser import SimpleBrowser
from modules.agent.memory.base_memory import BaseMemory
from modules.logger import BaseAgentLogger

class ResumActSearchAgent(BaseLLMAgent):
    """
    A ResumAct-based search agent that uses a Thought-Summarize-Action-Observation loop.
    It utilizes SimpleBrowser for web searching and page navigation.
    The 'Summarize' step creates a rubric-based report of current knowledge before taking action.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[BaseAgentLogger] = None,
        max_conversational_turns: Optional[int] = None,
    ):
        """
        Initialize the ResumAct Search Agent.

        Args:
            config: Configuration for the LLM sampler.
            browser_config: Configuration for the browser tool.
            system_prompt: Optional custom system prompt.
            tools: Optional custom tool definitions.
            logger: Optional logger for tracking progress.
            max_conversational_turns: Optional limit on the number of conversational turns to keep in context.
        """
        super().__init__(
            config=config,
            system_prompt=system_prompt,
            tools=tools
        )

        # If prompts/tools were not loaded via agent_spec in super().__init__ 
        # and not explicitly provided, load from default prompts.yaml
        if not self.system_prompt or not self.tools:
            default_prompts = self._load_default_prompts()
            
            if not self.tools:
                self.tools = default_prompts.get("tools", [])
            
            if not self.system_prompt:
                current_date = datetime.now().strftime("%Y-%m-%d")
                tools_description = json.dumps(self.tools, indent=2, ensure_ascii=False)
                system_prompt_template = default_prompts.get("resumact_system_prompt", "")
                if system_prompt_template:
                    self.system_prompt = system_prompt_template.format(
                        tools_description=tools_description, 
                        current_date=current_date
                    )
        
        self.browser = SimpleBrowser(config=browser_config)
        self.logger = logger
        self.max_conversational_turns = max_conversational_turns

    def _load_default_prompts(self) -> Dict[str, Any]:
        """Loads the default prompt and tool configuration for ResumAct Search Agent."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        return self._load_prompts(yaml_path)

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Interact with the browser tool or handle agent-specific tools.
        """
        if tool_name == "finish":
            return tool_args.get("answer", "No answer provided.")
            
        # browser.do is synchronous
        response = self.browser.do(tool_name, **tool_args)
        return self.browser.format_response_as_text(response)

    async def run(self, user_query: str, max_steps: int = 15) -> AgentRunResult:
        """
        Runs the ResumAct loop to solve the user's query.
        Returns an AgentRunResult containing the final answer and the message trajectory.
        """
        # Initialize memory with the system prompt (full trajectory for the final result)
        memory = BaseMemory(system_prompt=self.system_prompt)
        memory.add_user_message(user_query)
        
        # Reduced context state
        last_assistant_msg = None # SamplerResponse
        last_tool_messages = []   # List of tool message dicts
        
        # Initialize tool call counts
        tool_counts = {}
        
        error_msg = None
        for step in range(max_steps):
            # Construct the reduced context for this turn:
            # 1. System Prompt (via generate_response include_system=True)
            # 2. Original Question (User)
            current_messages = [{"role": "user", "content": user_query}]
            
            if step > 0:
                # 3. Previous Turn Assistant Message
                if last_assistant_msg:
                    assistant_msg = {
                        "role": "assistant",
                        "content": last_assistant_msg.content,
                    }
                    if last_assistant_msg.reasoning_content:
                        assistant_msg["reasoning_content"] = last_assistant_msg.reasoning_content
                    if last_assistant_msg.tool_calls:
                        assistant_msg["tool_calls"] = last_assistant_msg.tool_calls
                    current_messages.append(assistant_msg)
                
                # 4. Previous Turn Tool Observations
                if last_tool_messages:
                    current_messages.extend(last_tool_messages)
                
            
            # 5. Mandatory prompt at the end of context to ensure compliance
            if error_msg:   
                current_messages.append({
                    "role": "user", 
                    "content": error_msg
                })
            else:
                current_messages.append({
                    "role": "user", 
                    "content": "Please provide your Thought, the Knowledge Rubric in <report> tags, and your next Action(s)."
                })

            # Generate response from LLM
            # We use current_messages instead of the full memory history
            response = self.generate_response(
                current_messages,
                include_system=True
            )
            content = response.content
            tool_calls = response.tool_calls or []

            # Reset turn-specific state for context tracking
            last_tool_messages = []

            # Log the assistant's message in the full memory
            memory.add_assistant_message(
                content=content,
                reasoning_content=response.reasoning_content,
                tool_calls=tool_calls
            )

            # Check if <report> tags exist in content
            if "<report>" not in content or "</report>" not in content:
                if self.logger:
                    self.logger.log_agent_result(self.__class__.__name__, f"Step {step} error: Missing <report>", False)
                error_msg = "Error: Your response is missing the required <report> block. Please provide the Knowledge Rubric report in every turn."
                continue
            else:
                error_msg = None
                # Extract and print the report for visibility
                report_match = re.search(r"<report>(.*?)</report>", content, re.DOTALL)
                if report_match:
                    report_content = report_match.group(1).strip()
                    print(f"\n{'='*20} Step {step} Knowledge Rubric {'='*20}\n{report_content}\n{'='*65}\n")

                if self.logger:
                    # Log the content (which should contain the report)
                    self.logger.log_agent_result(self.__class__.__name__, f"Step {step} content:\n{content}", True)

            
            last_assistant_msg = response 
            if not tool_calls:
                # If no valid action, it might just be a thought-only response
                continue

            # Execute all tool calls found in the response
            for tool_call in tool_calls:
                # Handle both formats: {name, arguments} and {function: {name, arguments}}
                if "function" in tool_call:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                else:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except:
                        pass
                
                tool_call_id = tool_call.get("id", f"call_{step}_{tool_name}")

                # Track tool counts
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

                if self.logger:
                    self.logger.log_agent_tool_execution(self.__class__.__name__, tool_name, tool_args)

                if tool_name == "finish":
                    answer = tool_args.get("answer", content)
                    return AgentRunResult(
                        answer=answer,
                        trajectory=memory.get_messages(),
                        status="success",
                        metadata={
                            "tool_counts": tool_counts,
                            "total_tool_calls": sum(tool_counts.values())
                        }
                    )

                # Execute tool and get observation
                try:
                    observation = await self.call_tool(tool_name, tool_args)
                except Exception as e:
                    observation = f"Error executing {tool_name}: {str(e)}"

                # Add tool observation to both full memory and reduced context tracking
                obs_content = f"Observation: {observation}"
                memory.add_tool_message(tool_call_id, obs_content, tool_name=tool_name)
                last_tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "content": obs_content
                })

                if self.logger:
                    # Keep it as a result-style line for visibility; not marking failure here
                    # because the observation itself may contain tool errors.
                    self.logger.log_agent_result(self.__class__.__name__, f"{tool_name} -> {observation}", True)
        
        return AgentRunResult(
            answer="Failed to reach a final answer within the maximum number of steps.",
            trajectory=memory.get_messages(),
            status="max_steps_reached",
            metadata={
                "tool_counts": tool_counts,
                "total_tool_calls": sum(tool_counts.values())
            }
        )
        
        return AgentRunResult(
            answer="Failed to reach a final answer within the maximum number of steps.",
            trajectory=memory.get_messages(),
            status="max_steps_reached",
            metadata={
                "tool_counts": tool_counts,
                "total_tool_calls": sum(tool_counts.values())
            }
        )
