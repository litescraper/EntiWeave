import os
import re
from typing import Optional, Dict, Any, List

import asyncio
from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, AgentRunResult, MessageList
from modules.agent.memory.base_memory import BaseMemory
from modules.graph.knowledge_graph.cokg import CompositionalKnowledgeGraph

class Graph2QueryAgent(BaseLLMAgent):
    """
    Agent that converts a Compositional Knowledge Graph (CoKG) into a coherent question.
    """
    
    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        logger: Optional[Any] = None,
        prompts_path: str = os.path.join(os.path.dirname(__file__), "prompts.yaml")
    ):
        # Load prompts from yaml
        prompts = self._load_prompts(prompts_path)
        system_prompt = prompts.get("graph2query_system_prompt", "")
        self.user_prompt_template = prompts.get("graph2query_user_prompt", "")
        
        # Adjust max_tokens to accommodate long reasoning if not explicitly set high
        if config:
            if "max_tokens" not in config.options:
                config.options["max_tokens"] = 16384
            else:
                # Ensure it's at least 16384 if the user set it lower
                config.options["max_tokens"] = max(config.options["max_tokens"], 16384)
            
        super().__init__(config=config, system_prompt=system_prompt)
        self.logger = logger

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
                info += f"Name: '{entity.name}'"
            else:
                info += f"Name: '{entity.name}'"
            
            # Find predicates where this entity is the input
            child_preds = [p for p in graph.predicates.values() if p.input == entity_id]
            if child_preds:
                for p in child_preds:
                    info += "\n" + f"{prefix}  * Relationship {p.id}: {p.content}"
                    for out_id in p.outputs:
                        info += "\n" + describe_entity(out_id, indent + 4)
            return info

        return describe_entity("E0")

    async def run(self, graph: CompositionalKnowledgeGraph) -> AgentRunResult:
        """
        Generates a question from the provided CoKG.
        """
        if "E0" not in graph.entities:
            raise ValueError("The provided CoKG must contain an entity with ID 'E0'.")
            
        e0 = graph.entities["E0"]
        graph_text = self._format_graph(graph)
        
        user_prompt = self.user_prompt_template.format(
            graph_text=graph_text,
            e0_name=e0.name,
            e0_description=e0.description
        )
        
        messages: MessageList = [
            {"role": "user", "content": user_prompt}
        ]
        
        max_retries = 5
        question = ""
        exact_answer = ""
        content = ""
        full_text = ""
        
        for attempt in range(max_retries):
            formatted_messages = BaseMemory.format_messages(messages, sampler_type=self.config.sampler_type, include_thinking=True)
            # Use to_thread because the sampler call (requests) is blocking
            response = await asyncio.to_thread(self.generate_response, formatted_messages)
            content = response.content.strip()
            reasoning = (response.reasoning_content or "").strip()
            
            # Combine content and reasoning to search for tags, 
            # as some models might put tags in reasoning or reasoning might be the only output
            full_text = f"{reasoning}\n{content}".strip()
            
            # Parse question and answer from tags (taking the last occurrence)
            question_matches = re.findall(r"<question>(.*?)</question>", full_text, re.DOTALL)
            answer_matches = re.findall(r"<answer>(.*?)</answer>", full_text, re.DOTALL)
            
            question = question_matches[-1].strip() if question_matches else ""
            exact_answer = answer_matches[-1].strip() if answer_matches else ""
            
            if question and exact_answer:
                return AgentRunResult(
                    answer=question,
                    trajectory=messages + [{"role": "assistant", "content": content, "reasoning_content": reasoning}],
                    status="success",
                    metadata={
                        "e0_name": e0.name,
                        "e0_description": e0.description,
                        "exact_answer": exact_answer,
                        "raw_content": content,
                        "reasoning": reasoning
                    }
                )
            
            if self.logger:
                self.logger.log_agent_result("Graph2Query", f"Attempt {attempt + 1} failed: missing or empty <question> or <answer> tags.", False)
            
            messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
            messages.append({"role": "user", "content": "Your response MUST include both <question> and <answer> tags, and neither can be empty. Please ensure the question reflects all constraints in the CoKG. \n\nIMPORTANT: If your reasoning is very long, please STOP reasoning and output the <question> and <answer> tags immediately. The final output must contain these tags."})

        # Fallback if all retries fail
        final_question = question if question else full_text
        final_exact_answer = exact_answer if exact_answer else e0.name
        
        return AgentRunResult(
            answer=final_question,
            trajectory=messages,
            status="error" if not (final_question and final_exact_answer) else "partial_success",
            metadata={
                "e0_name": e0.name,
                "e0_description": e0.description,
                "exact_answer": final_exact_answer,
                "raw_content": content,
                "reasoning": reasoning,
                "error": "Failed to generate valid question/answer after max retries."
            }
        )
