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
from modules.agent.python.python_interpreter import PythonInterpreter
from modules.agent.tools.tool_mapping import get_tool_classes
from modules.agent.memory.base_memory import BaseMemory
from modules.agent.memory.context_strategies import get_context_strategy
from modules.logger import BaseAgentLogger

class ReactSearchAgent(BaseLLMAgent):
    """
    A ReAct-based search agent that uses a Thought-Action-Observation loop.
    It utilizes SimpleBrowser for web searching and page navigation.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[BaseAgentLogger] = None,
        max_conversational_turns: Optional[int] = None,
        max_tool_context: Optional[int] = None,
        reset_interval: Optional[int] = None,
        context_strategy_name: str = "sliding_window",
        tokenizer_path: Optional[str] = None,
        max_tokens: Optional[int] = 110000,
    ):
        """
        Initialize the ReAct Search Agent.

        Args:
            config: Configuration for the LLM sampler.
            browser_config: Configuration for the browser tool.
            system_prompt: Optional custom system prompt.
            tools: Optional custom tool definitions.
            logger: Optional logger for tracking progress.
            max_conversational_turns: Optional limit on the number of conversational turns to keep in context.
            max_tool_context: Optional limit on the number of recent tool results to keep uncompressed.
            reset_interval: Optional number of turns after which to reset all assistant-tool messages.
            context_strategy_name: Strategy for context management ("sliding_window" or "token_limit").
            tokenizer_path: Path to the tokenizer (required if using "token_limit").
            max_tokens: Maximum tokens allowed in context (used if strategy is "token_limit").
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
                # Add python tool if available in default prompts
                python_tool = default_prompts.get("code_tool", [])
                if python_tool:
                    self.tools.extend(python_tool)
            
            if not self.system_prompt:
                current_date = datetime.now().strftime("%Y-%m-%d")
                system_prompt_template = default_prompts.get("react_system_prompt", "")
                if system_prompt_template:
                    self.system_prompt = system_prompt_template.format(current_date=current_date)

        # Tool classes for modular tools
        tool_classes = get_tool_classes()
        
        # Determine required tool names from self.tools
        tool_names = set()
        for tool in self.tools:
            if "function" in tool:
                tool_names.add(tool["function"]["name"])
            else:
                tool_names.add(tool.get("name"))
        
        # Shared resources (initialized only if needed)
        self.browser = None
        if any(name in ["search", "visit", "open", "click", "retrieve"] for name in tool_names):
            self.browser = SimpleBrowser(config=browser_config)
            
        self.python_interpreter = None
        if "python" in tool_names:
            self.python_interpreter = PythonInterpreter()
            
        # Mapping between tool name and initialized tool instance
        self.tool_instances = {}
        for name in tool_names:
            if name in tool_classes:
                if name in ["search", "visit", "open", "click", "retrieve"]:
                    # For VisitPageTool (used for both 'visit' and 'open'), pass the specific name
                    if name in ["visit", "open"]:
                        self.tool_instances[name] = tool_classes[name](name=name)
                    else:
                        self.tool_instances[name] = tool_classes[name]()
                else:
                    self.tool_instances[name] = tool_classes[name]()
        
        self.logger = logger
        
        # Initialize context strategy
        self.context_strategy = get_context_strategy(
            context_strategy_name,
            max_conversational_turns=max_conversational_turns,
            max_tool_context=max_tool_context,
            reset_interval=reset_interval,
            tokenizer_path=tokenizer_path,
            max_tokens=max_tokens,
            logger=self.logger
        )

    def _load_default_prompts(self) -> Dict[str, Any]:
        """Loads the default prompt and tool configuration for ReAct Search Agent."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        return self._load_prompts(yaml_path)

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any], query: Optional[str] = None) -> str:
        """
        Interact with the mapped tools or handle agent-specific tools.
        """
        if tool_name in ["finish", "answer_attempt"]:
            return f"Observation: Attempted Answer: {tool_args.get('answer', 'No answer provided.')}"

        if tool_name in self.tool_instances:
            # Prepare arguments for the tool call
            kwargs = tool_args.copy()
            
            # Compatibility with Tongyi
            if tool_name == "visit" and "goal" in kwargs:
                tool_name = "retrieve"
                kwargs["query"] = kwargs.pop("goal")
            ### temporary to enforce the goal is always present
            if tool_name == "visit" and "goal" not in kwargs:
                kwargs["goal"] = f"Get relevant information that can be used to answer this question: {query}"

            if tool_name in ["search", "visit", "open", "click", "retrieve"]:
                kwargs["browser"] = self.browser
                if tool_name == "retrieve":
                    kwargs["sampler"] = self.sampler
            elif tool_name == "python":
                kwargs["interpreter"] = self.python_interpreter
            
            # Each tool is expected to have a 'call' method
            return await self.tool_instances[tool_name].call(**kwargs)
        
        return f"Error: Tool {tool_name} not found. Use only valid tools available in the tools description!"

    def _get_truncated_messages(self, messages: MessageList) -> MessageList:
        """
        Truncates the message history based on the configured context strategy.
        """
        return self.context_strategy.truncate_messages(messages)

    async def run(self, user_query: str, max_steps: int = 15) -> AgentRunResult:
        """
        Runs the ReAct loop to solve the user's query.
        Returns an AgentRunResult containing the final answer and the message trajectory.
        """
        # Initialize memory with the system prompt
        memory = BaseMemory(system_prompt=self.system_prompt)
        memory.add_user_message(user_query)
        
        # Initialize tool call counts and trajectories
        tool_counts = {}
        tool_trajectories = []
        
        for step in range(max_steps):
            # Get current messages
            messages = memory.get_messages()
            
            # 1. Check for termination BEFORE generating response
            # This handles cases where a terminal prompt was injected in the PREVIOUS turn.
            if self.context_strategy.check_termination(messages):
                if self.logger:
                    self.logger.warning("Token limit reached and forced termination after one final turn. Returning results.")
                return AgentRunResult(
                    answer="Terminated due to token limit after one final turn.",
                    trajectory=memory.get_messages(),
                    status="max_tokens_reached",
                    metadata={
                        "tool_counts": tool_counts,
                        "total_tool_calls": sum(tool_counts.values()),
                        "tool_trajectories": tool_trajectories
                    }
                )

            # 2. Check if context strategy wants to inject a terminal prompt (e.g., token limit reached)
            # OR if we are at the final step allowed by max_steps.
            terminal_prompt = self.context_strategy.get_terminal_prompt(messages)
            
            if not terminal_prompt and step == max_steps - 1:
                terminal_prompt = "This is the FINAL step. You must output call finish tool now. Do not search again."
            
            if terminal_prompt:
                memory.add_user_message(f"Observation: {terminal_prompt}")
                # Refresh messages after injection for truncation logic
                messages = memory.get_messages()

            # 3. Truncate and format messages for the sampler
            truncated_messages = self._get_truncated_messages(messages)
            
            # Format messages for the sampler
            formatted_messages = BaseMemory.format_messages(
                truncated_messages, 
                sampler_type=self.config.sampler_type, 
                include_thinking=True
            )

            # Generate response from LLM
            # We pass include_system=False because it's already in the memory/truncated messages
            response = self.generate_response(
                formatted_messages, 
                include_system=False
            )
            content = response.content
            print(response)
            
            # Use tool calls directly from response
            tool_calls = response.tool_calls or []
            
            # Log the assistant's thought/action in memory
            memory.add_assistant_message(
                content=content,
                reasoning_content=response.reasoning_content,
                tool_calls=tool_calls
            )
            
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
                
                tool_call_id = tool_call.get("id", f"call_{step}")

                # Track tool counts
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

                if self.logger:
                    self.logger.log_agent_tool_execution(self.__class__.__name__, tool_name, tool_args)

                if tool_name == "finish":
                    answer = tool_args.get("answer", content)
                    return AgentRunResult(
                        answer=str(answer),
                        trajectory=memory.get_messages(),
                        status="success",
                        metadata={
                            "tool_counts": tool_counts,
                            "total_tool_calls": sum(tool_counts.values()),
                            "tool_trajectories": tool_trajectories
                        }
                    )

                # Execute tool and get observation
                try:
                    observation = await self.call_tool(tool_name, tool_args, query=user_query)
                    
                    # Capture tool metadata (e.g., internal LLM trajectories)
                    # Handle internal tool name switching (e.g., visit with goal -> retrieve)
                    lookup_name = tool_name
                    if tool_name == "visit" and isinstance(tool_args, dict) and ("goal" in tool_args or "query" in tool_args):
                        lookup_name = "retrieve"

                    if lookup_name in self.tool_instances:
                        tool_instance = self.tool_instances[lookup_name]
                        if hasattr(tool_instance, "metadata") and tool_instance.metadata:
                            tool_trajectories.append({
                                "step": step,
                                "tool_name": lookup_name,
                                "tool_args": tool_args,
                                "metadata": tool_instance.metadata.copy()
                            })
                            # Reset metadata for next tool call
                            tool_instance.reset_metadata()

                    # Add tool observation to memory
                    memory.add_tool_message(tool_call_id, f"Observation: {observation}", tool_name=tool_name)
                except Exception as e:
                    observation = f"Error executing {tool_name}: {str(e)}"
                    memory.add_tool_message(
                        tool_call_id, 
                        f"Observation: {observation}",
                        tool_name=tool_name
                    )

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
                "total_tool_calls": sum(tool_counts.values()),
                "tool_trajectories": tool_trajectories
            }
        )

