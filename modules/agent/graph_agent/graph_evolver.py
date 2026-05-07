import os
import re
import json
import logging
import uuid
import copy
from typing import Optional, Dict, Any, List
from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, AgentRunResult, MessageList
from modules.agent.memory.base_memory import BaseMemory
from modules.graph.knowledge_graph.cokg import CompositionalKnowledgeGraph
from modules.graph.database import DatabaseManager

class GraphEvolver(BaseLLMAgent):
    """
    Agent that refines and prunes a Compositional Knowledge Graph (CoKG) to encourage deep search.
    """
    
    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        graph: Optional[CompositionalKnowledgeGraph] = None,
        logger: Optional[Any] = None,
        prompts_path: str = os.path.join(os.path.dirname(__file__), "prompts.yaml"),
        run_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        seed_object_id: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        # Load prompts from yaml
        prompts = self._load_prompts(prompts_path)
        system_prompt = prompts.get("graph_evolver_system_prompt", "")
        self.tools_schema = prompts.get("graph_evolver_tools", [])
        
        super().__init__(config=config, system_prompt=system_prompt, tools=self.tools_schema)
        self.graph = graph
        self.logger = logger or logging.getLogger(__name__)
        self.run_id = run_id
        self.graph_id = graph_id
        self.seed_object_id = seed_object_id
        self.db_manager = db_manager

    def _format_graph(self, graph: CompositionalKnowledgeGraph) -> str:
        """
        Formats the CoKG into a nested paragraph-style representation for the LLM.
        """
        visited = set()

        def describe_entity(entity_id, indent=0):
            if entity_id in visited:
                return f"{'  ' * indent}- {entity_id} (already described above)"
            visited.add(entity_id)
            
            entity = graph.entities.get(entity_id)
            if not entity:
                return ""
            
            prefix = "  " * indent
            info = f"{prefix}- Entity {entity_id} (Category: {entity.category}): "
            if entity_id == "E0":
                info += f"Description: '{entity.description}', Name: '{entity.name}'"
            else:
                info += f"Name: '{entity.name}', Description: '{entity.description}'"
            
            # Find predicates where this entity is the input
            child_preds = [p for p in graph.predicates.values() if p.input == entity_id]
            if child_preds:
                for p in child_preds:
                    info += "\n" + f"{prefix}  * Relationship {p.id}: {p.content}"
                    for out_id in p.outputs:
                        info += "\n" + describe_entity(out_id, indent + 4)
            return info

        return describe_entity("E0")

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        if tool_name == "evolve_predicate":
            # Backup the graph to allow rollback if prune limit is exceeded
            graph_backup = copy.deepcopy(self.graph)
            result = self._evolve_predicate(
                tool_args.get("predicate_id"), 
                tool_args.get("new_content")
            )
            if self._check_prune_limit_exceeded():
                self.graph = graph_backup
                limit = int(self.initial_predicate_count * 0.2)
                return f"Error: Evolving this predicate would result in too many predicates being pruned (exceeding the 20% limit of {limit}). Please keep the predicate content more similar to the original to preserve connectivity."
            return result
        elif tool_name == "prune_predicate":
            # Backup the graph
            graph_backup = copy.deepcopy(self.graph)
            result = self._prune_predicate(tool_args.get("predicate_id"))
            if self._check_prune_limit_exceeded():
                self.graph = graph_backup
                limit = int(self.initial_predicate_count * 0.2)
                return f"Error: Pruning this predicate would exceed the 20% limit of the whole predicates ({limit} predicates). Please use evolve_predicate instead to make it more fuzzy without removing it."
            return result
        elif tool_name == "finish":
            return {"status": "finished", "reasoning": tool_args.get("reasoning")}
        else:
            return f"Error: Tool {tool_name} not found."

    def _check_prune_limit_exceeded(self) -> bool:
        """Checks if the current number of pruned predicates exceeds the 20% limit."""
        if not hasattr(self, 'initial_predicate_count') or self.initial_predicate_count == 0:
            return False
        
        current_count = len(self.graph.predicates)
        pruned_count = self.initial_predicate_count - current_count
        # Limit is 20% of whole predicates
        limit = self.initial_predicate_count * 0.2
        return pruned_count > limit

    def _evolve_predicate(self, predicate_id: str, new_content: str) -> str:
        if predicate_id not in self.graph.predicates:
            return f"Error: Predicate {predicate_id} not found."
        
        predicate = self.graph.predicates[predicate_id]
        old_content = predicate.content
        
        # Extract all entity placeholders from the new content
        # Placeholders are like <|-E1-|> or <|-C1-|>
        new_entity_ids = set(re.findall(r"<|-([\w@]+)-|>", new_content))
        
        # Identify which outputs were removed
        old_outputs = list(predicate.outputs)
        new_outputs = [eid for eid in old_outputs if eid in new_entity_ids]
        removed_outputs = [eid for eid in old_outputs if eid not in new_entity_ids]
        
        # Update predicate
        predicate.content = new_content
        predicate.outputs = new_outputs
        
        pruned_info = [f"Successfully evolved predicate {predicate_id} from '{old_content}' to '{new_content}'."]
        
        # If any outputs were removed, they might be orphaned
        for entity_id in removed_outputs:
            pruned_info.extend(self._prune_entity_if_orphaned(entity_id))
            
        return "\n".join(pruned_info)

    def _prune_predicate(self, predicate_id: str) -> str:
        if predicate_id not in self.graph.predicates:
            return f"Error: Predicate {predicate_id} not found."
        
        predicate = self.graph.predicates[predicate_id]
        outputs = list(predicate.outputs)
        
        # Remove the predicate itself
        del self.graph.predicates[predicate_id]
        
        # Check downstream entities and predicates
        pruned_info = [f"Pruned predicate {predicate_id}"]
        for entity_id in outputs:
            pruned_info.extend(self._prune_entity_if_orphaned(entity_id))
            
        return "\n".join(pruned_info)

    def _prune_entity_if_orphaned(self, entity_id: str) -> List[str]:
        if entity_id == "E0":
            return []
        
        # An entity is orphaned if it's not the root (E0) 
        # AND it's not an output of any remaining predicate.
        is_output_of_any = False
        for p in self.graph.predicates.values():
            if entity_id in p.outputs:
                is_output_of_any = True
                break
        
        pruned_items = []
        if not is_output_of_any:
            # Find and prune predicates where this entity is the input
            # We do this BEFORE deleting the entity so we have its ID
            preds_to_prune = [pid for pid, p in self.graph.predicates.items() if p.input == entity_id]
            
            # Prune this entity
            if entity_id in self.graph.entities:
                del self.graph.entities[entity_id]
                pruned_items.append(f"Pruned orphaned entity {entity_id}")
                
            for pid in preds_to_prune:
                # Recursively prune
                pruned_items.append(self._prune_predicate(pid))
        
        return pruned_items

    async def run(self, graph: Optional[CompositionalKnowledgeGraph] = None, max_turns: int = 5) -> AgentRunResult:
        if graph:
            self.graph = graph
            
        if not self.graph:
            raise ValueError("No graph provided to GraphEvolver.")

        self.initial_predicate_count = len(self.graph.predicates)

        # Initialize memory with the system prompt
        memory = BaseMemory(system_prompt=self.system_prompt)
        
        # Prepare the initial state for the agent
        graph_text = self._format_graph(self.graph)
        user_message = f"Please refine and evolve the following CoKG to encourage deep search:\n\n{graph_text}"

        memory.add_user_message(user_message)
        
        status = "in_progress"
        final_answer = ""
        
        for turn in range(max_turns):
            # Generate response from LLM
            # We pass include_system=False because it's already in the memory
            formatted_messages = memory.get_formatted_messages(
                sampler_type=self.config.sampler_type, 
                include_thinking=True
            )
            response = self.generate_response(formatted_messages, include_system=False)
            content = response.content
            
            # Use tool calls directly from response
            tool_calls = response.tool_calls or []
            
            memory.add_assistant_message(
                content=content,
                reasoning_content=response.reasoning_content,
                tool_calls=tool_calls
            )
            
            if not tool_calls:
                # Check if the agent just finished without tool call
                if "finish" in content.lower() or "done" in content.lower():
                    status = "success"
                    final_answer = content
                    break
                
                # If no tool calls, ask to use tools
                retry_msg = "Please use the available tools to refine and evolve the graph or call finish if you are done."
                memory.add_user_message(retry_msg)
                continue

            # Execute tool calls
            tool_results = []
            finished = False
            for tc in tool_calls:
                tc_function = tc["function"]
                tool_args = tc_function["arguments"]
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        pass
                # Log tool execution
                if self.logger:
                    self.logger.log_agent_tool_execution("GraphEvolver", tc_function["name"], tool_args)

                result = await self.call_tool(tc_function["name"], tool_args)
                tool_results.append(result)
                
                if tc_function["name"] == "finish":
                    finished = True
                    if isinstance(result, dict):
                        final_answer = result.get("reasoning", "")
                    else:
                        final_answer = str(result)
                    status = "success"
                    # We continue to process other tools in the same call if any
            
            # Format tool results back to LLM
            result_content = ""
            for i, (tc, res) in enumerate(zip(tool_calls, tool_results)):
                tc_id = tc.get("id", f"call_{i}")
                result_content += f"Result of {tc_function['name']} ({tc_id}):\n{res}\n\n"
            
            memory.add_user_message(result_content.strip())
            
            if finished:
                break
        else:
            status = "max_steps_reached"
            final_answer = "Max steps reached without explicit finish."

        return AgentRunResult(
            answer=final_answer,
            trajectory=memory.get_messages(),
            status=status,
            metadata={"graph": self.graph.to_dict()}
        )
