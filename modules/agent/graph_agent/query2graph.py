import os
import json
import asyncio
from typing import List, Dict, Any, Optional

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, SamplerResponse, AgentRunResult
from modules.agent.memory.base_memory import BaseMemory
from modules.graph.knowledge_graph.cokg import CompositionalKnowledgeGraph
from .utils import format_tools_description

class Query2GraphAgent(BaseLLMAgent):
    """
    An agent that translates natural language queries into a Compositional Knowledge Graph (CoKG)
    using a Reverse Construction Strategy.
    
    It decomposes complex multi-hop questions starting from the target entity (the answer)
    and recursively expanding into its dependencies.
    
    CoKG is a directed acyclic graph G = (V_E ∪ V_P, E), where:
    - V_E is the set of entities (Constants or Variables).
    - V_P is the set of predicates (p: Ein -> Eouts).
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the Query2GraphAgent.
        """
        # Load prompts and tools from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        prompt_data = self._load_prompts(yaml_path)

        # Use tools from yaml if not provided
        effective_tools = tools or prompt_data.get("query2graph_tools", [])
        
        # Use system prompt from yaml if not provided
        if not system_prompt:
            tools_description = format_tools_description(effective_tools)
            system_prompt_template = prompt_data.get("query2graph_system_prompt", "")
            effective_system_prompt = system_prompt_template.format(tools_description=tools_description)
        else:
            effective_system_prompt = system_prompt
        
        super().__init__(
            config=config,
            system_prompt=effective_system_prompt,
            tools=effective_tools
        )
        
        self.cokg = CompositionalKnowledgeGraph()

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Executes tools to build the CoKG incrementally.
        """
        if tool_name == "add_entity":
            try:
                is_var = tool_args.get("is_variable", True)
                if isinstance(is_var, str):
                    is_var = is_var.lower() == "true"
                
                # Support both old 'content' and new 'name'/'description'
                name = tool_args.get("name")
                description = tool_args.get("description")
                content = tool_args.get("content")
                
                if not name and content:
                    name = content
                if not description and content:
                    description = content
                
                if not name:
                    name = "Unknown"
                if not description:
                    description = "No description provided"

                entity = self.cokg.add_entity(
                    id=tool_args.get("id"),
                    name=name,
                    description=description,
                    category=tool_args.get("category", "Unknown"),
                    is_variable=is_var,
                    links=tool_args.get("links", [])
                )
                return f"Successfully added entity: {repr(entity)}"
            except Exception as e:
                return f"Error adding entity: {str(e)}"
        
        elif tool_name == "add_predicate":
            try:
                outputs = tool_args["outputs"]
                if isinstance(outputs, str):
                    try:
                        outputs = json.loads(outputs)
                    except:
                        # Fallback for comma separated or single string
                        if outputs.startswith("[") and outputs.endswith("]"):
                            outputs = outputs[1:-1].replace("'", "").replace('"', "").split(",")
                            outputs = [o.strip() for o in outputs if o.strip()]
                        else:
                            outputs = [outputs.strip()]

                predicate = self.cokg.add_predicate(
                    id=tool_args.get("id"),
                    content=tool_args["content"],
                    input=tool_args["input"],
                    outputs=outputs,
                    links=tool_args.get("links", [])
                )
                return f"Successfully added predicate: {repr(predicate)}"
            except ValueError as e:
                return f"Validation Error: {str(e)}"
            except Exception as e:
                return f"Error adding predicate: {str(e)}"
        
        elif tool_name == "finish":
            reasoning = tool_args.get("reasoning", "No reasoning provided.")
            return f"Graph construction complete. Reasoning: {reasoning}"
        
        else:
            return f"Unknown tool: {tool_name}"

    async def run(self, query: str, max_steps: int = 30) -> AgentRunResult:
        """
        Runs the agent loop to translate the query into a CoKG using Reverse Construction.
        The agent starts by identifying the target entity and then decomposes it into dependencies.
        Returns an AgentRunResult containing the CoKG as a JSON string and the trajectory.
        """
        # Reset CoKG for a new run
        self.cokg = CompositionalKnowledgeGraph()
        
        # Initialize memory with the system prompt
        memory = BaseMemory(system_prompt=self.system_prompt)
        memory.add_user_message(
            f"Please translate the following query into a CoKG using the Reverse Construction Strategy:\n\n{query}"
        )
        
        for step in range(max_steps):
            # Generate response from LLM
            # We pass include_system=False because it's already in the memory
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
                # If no valid action, ask the LLM to use tools
                memory.add_user_message(
                    "Observation: No valid tool call found. Please use the available tools to build the graph or 'finish' to complete."
                )
                continue

            # Execute tool calls
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_call_id = tool_call.get("id", f"call_{step}")

                # Handle finish tool
                if tool_name == "finish":
                    await self.call_tool(tool_name, tool_args)
                    return AgentRunResult(
                        answer=json.dumps(self.cokg.to_dict(), indent=2, ensure_ascii=False),
                        trajectory=memory.get_messages(),
                        status="success"
                    )

                # Execute other tools and get observation
                try:
                    observation = await self.call_tool(tool_name, tool_args)
                    memory.add_tool_message(tool_call_id, f"Observation: {observation}", tool_name=tool_name)
                except Exception as e:
                    memory.add_tool_message(
                        tool_call_id, 
                        f"Observation: Error executing {tool_name}: {str(e)}",
                        tool_name=tool_name
                    )
        
        # If max steps reached, return what we have
        return AgentRunResult(
            answer=json.dumps(self.cokg.to_dict(), indent=2, ensure_ascii=False),
            trajectory=memory.get_messages(),
            status="max_steps_reached"
        )
