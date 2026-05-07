import os
import json
import re
import uuid
import random
from typing import List, Dict, Any, Optional, Tuple

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, AgentRunResult
from modules.agent.memory.base_memory import BaseMemory
from modules.graph.knowledge_graph.cokg import Entity, CompositionalKnowledgeGraph
from modules.logger import BaseAgentLogger
from .utils import format_predicates


class PredicateSamplerAgent(BaseLLMAgent):
    """
    An agent that samples predicates from knowledge search results and builds the knowledge graph.
    It selects minimal but informative predicates and extracts entities for further expansion.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        system_prompt: Optional[str] = None,
        logger: Optional[BaseAgentLogger] = None,
    ):
        # Load prompts and tools from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        prompt_data = self._load_prompts(yaml_path)

        super().__init__(
            config=config, 
            system_prompt="", 
            tools=prompt_data.get("predicate_sampler_tools", [])
        )

        self.prompt_data = prompt_data
        self.link_mapping = {}
        self.logger = logger

    def _format_existing_entities(self, existing_entities: List[Dict[str, Any]]) -> str:
        """Format existing entities information for context."""
        if not existing_entities:
            return ""

        formatted_entities = []
        for entity in existing_entities:
            entity_id = entity.get("id", "")
            name = entity.get("name", "")
            description = entity.get("description", "")

            # Truncate description to first several words (around 10-15 words max)
            words = description.split()
            truncated_desc = " ".join(words[:8]) + ("..." if len(words) > 8 else "")

            formatted_entities.append(f"- {entity_id}: {name} ({truncated_desc})")

        return "\n".join(formatted_entities)

    # Required parameters per tool for validation
    _TOOL_REQUIRED_PARAMS: Dict[str, List[str]] = {
        "add_entity": ["id", "name", "description", "category"],
        "add_predicate": ["content", "input", "outputs"],
    }

    @staticmethod
    def _normalize_output_ids(outputs_raw: Any) -> List[str]:
        """Normalize outputs from tool args (list or string like '[E1, E2]') to list of entity id strings."""
        if not outputs_raw:
            return []
        if isinstance(outputs_raw, list):
            return [str(o).strip() for o in outputs_raw if str(o).strip()]
        if isinstance(outputs_raw, str):
            return [s.strip() for s in outputs_raw.strip("[]").replace(",", " ").split() if s.strip()]
        return []

    @staticmethod
    def _entity_token_for(content: str, entity_id: str) -> str:
        """Return the token form that must appear in predicate content for entity_id (e.g. E1 -> <|-E1-|>)."""
        return f"<|-{entity_id}-|>"

    @staticmethod
    def _predicate_output_tokens_validation(content: str, output_ids: List[str]) -> List[str]:
        """
        Check that every entity in outputs has its token (<|-Ei-|>) inside content.
        Returns list of output entity ids that are missing their token in content.
        """
        if not content or not output_ids:
            return list(output_ids) if output_ids else []
        content_str = str(content).strip()
        missing: List[str] = []
        for eid in output_ids:
            eid_str = str(eid).strip()
            if not eid_str:
                continue
            token = PredicateSamplerAgent._entity_token_for(content_str, eid_str)
            if token not in content_str:
                missing.append(eid_str)
        return missing

    @staticmethod
    def _parse_tool_call(call: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract tool name and parsed arguments from a tool call (supports function wrapper or flat keys)."""
        name = None
        if "function" in call:
            name = call["function"].get("name")
            raw = call["function"].get("arguments")
        else:
            name = call.get("name")
            raw = call.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                args = {}
        else:
            args = raw or {}
        return (name, args)

    def _validate_tool_calls(self, parsed_calls: List[Tuple[Optional[str], Dict[str, Any]]]) -> List[str]:
        """
        Validate parsed tool calls. Returns empty list if at least one add_entity 
        and 1-4 add_predicates are valid, even if other calls are broken.
        """
        errors: List[str] = []

        if not parsed_calls:
            return ["No tool calls provided. You MUST use both 'add_entity' and 'add_predicate' tools."]

        valid_entity_count = 0
        valid_predicate_count = 0
        
        for i, (name, args) in enumerate(parsed_calls):
            if not name:
                continue
            
            # Check required parameters for this tool call
            required = self._TOOL_REQUIRED_PARAMS.get(name)
            call_is_valid = True
            if required:
                missing = [p for p in required if p not in args or not str(args.get(p, "")).strip()]
                if missing:
                    call_is_valid = False
                    errors.append(f"Tool '{name}' call {i+1} is missing required parameters: {', '.join(missing)}")
            
            if call_is_valid:
                if name == "add_entity":
                    valid_entity_count += 1
                elif name == "add_predicate":
                    valid_predicate_count += 1

        # If we have at least one valid entity and at least 1 valid predicate, we are good to go
        if valid_entity_count >= 1 and valid_predicate_count >= 1:
            return []
            
        # Otherwise, return consolidated errors
        final_errors = []
        if valid_entity_count < 1:
            final_errors.append("At least one valid 'add_entity' tool call is required.")
        if valid_predicate_count < 1:
            final_errors.append("At least one valid 'add_predicate' tool call is required.")
            
        return final_errors or errors

    def _prepare_prompts(
        self,
        target_entity: Entity,
        discovered_predicates: List[Dict[str, Any]],
        cokg: CompositionalKnowledgeGraph,
        parent_context: Optional[Dict[str, Any]] = None,
        existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, str]:
        """Prepare system and user prompts for the agent."""
        # Calculate next available entity ID
        existing_entity_ids = list(cokg.entities.keys())
        next_entity_id = self._get_next_entity_id(existing_entity_ids)
        target_entity_id = target_entity.id

        # Format predicates for the agent
        predicates_text, link_mapping = format_predicates(discovered_predicates, max_content_length=None)
        self.link_mapping = link_mapping

        # Ancestor Predicates Context
        ancestor_predicates_text = ""
        if parent_context and "ancestor_predicates" in parent_context:
            ancestor_predicates = parent_context["ancestor_predicates"]
            if ancestor_predicates:
                ancestor_predicates_text, _ = format_predicates(ancestor_predicates, max_content_length=300)
                ancestor_predicates_text = "\n\nAncestor Context (avoid creating duplicate/reverse predicates - includes all predicates from ancestor entities in the tree):\n" + ancestor_predicates_text

        # Existing Entities Context
        existing_entities_text = ""
        if existing_entities:
            formatted_entities = self._format_existing_entities(existing_entities)
            existing_entities_text = f"\n\nExisting Entities (avoid creating duplicate entities - check these before adding new ones):\n{formatted_entities}"
        
        # Create system prompt
        system_prompt_template = self.prompt_data.get("predicate_sampler_system_prompt", "")
        system_prompt = system_prompt_template.format(
            next_entity_id=next_entity_id,
            next_entity_id_plus_1=next_entity_id + 1,
            next_entity_id_plus_2=next_entity_id + 2,
            target_entity_id=target_entity_id
        )

        # Create user prompt
        user_prompt = f"""
Discovered Predicates ({len(discovered_predicates)} total):
{predicates_text}{ancestor_predicates_text}{existing_entities_text}

Please sample predicates that provide unique, distinguishing information about the target entity.
For each sampled predicate, extract entities that can be expanded for multi-hop reasoning.
Use the available tools, at least execute one add_entity and one add_predicate tool call in one turn (must provide 1-4 predicates)
"""
        return system_prompt, user_prompt

    def run(
        self,
        target_entity: Entity,
        discovered_predicates: List[Dict[str, Any]],
        cokg: CompositionalKnowledgeGraph,
        parent_context: Optional[Dict[str, Any]] = None,
        existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> AgentRunResult:
        """
        Sample predicates from discovered candidates and expand the knowledge graph.

        Args:
            target_entity: The entity we're expanding knowledge about
            discovered_predicates: List of predicate candidates from knowledge search
            cokg: The current knowledge graph to expand

        Returns:
            AgentRunResult with processing results and any new entities to explore
        """
        if not discovered_predicates:
            return AgentRunResult(
                status="no_predicates",
                answer="No predicates to sample from",
                trajectory=[],
                metadata={
                    "sampled_predicates": [],
                    "new_entities": []
                }
            )

        # 1. Prepare prompts
        system_prompt, user_prompt = self._prepare_prompts(
            target_entity=target_entity,
            discovered_predicates=discovered_predicates,
            cokg=cokg,
            parent_context=parent_context,
            existing_entities=existing_entities
        )
        self.system_prompt = system_prompt

        # 2. Run the agent with retry logic and unified validation
        messages = [{"role": "user", "content": user_prompt}]
        max_retries = 10
        retry_count = 0
        parsed_calls: List[Tuple[Optional[str], Dict[str, Any]]] = []
        content = ""
        validation_errors: List[str] = []

        while retry_count < max_retries:
            formatted_messages = BaseMemory.format_messages(messages, sampler_type=self.config.sampler_type, include_thinking=True)
            response = self.generate_response(formatted_messages, system=self.system_prompt, include_system=True)
            content = response.content.strip()

            if self.logger:
                self.logger.log_agent_result("PredicateSampler", f"Received response (Attempt {retry_count + 1})", True)
            
            current_tool_calls = response.tool_calls or []
            parsed_calls = [self._parse_tool_call(c) for c in current_tool_calls]

            # 3. Validation (uses pre-parsed calls to avoid parsing twice)
            validation_errors = self._validate_tool_calls(parsed_calls)
            if not validation_errors:
                break

            retry_count += 1
            error_msg = (
                f"Validation failed for your tool calls (Attempt {retry_count}/{max_retries}):\n"
                + "\n".join(f"- {err}" for err in validation_errors)
                + "\n\nPlease correct these errors and provide the FULL set of tool calls (entities and predicates) again. Ensure you include all required parameters."
            )
            if self.logger:
                val_error_list = "\n".join(f"- {err}" for err in validation_errors)
                self.logger.log_agent_result("PredicateSampler", f"Validation failed: {len(validation_errors)} errors: {val_error_list}", False)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": error_msg})

        if len(parsed_calls) == 0:
            return AgentRunResult(
                status="error",
                answer="Failed to generate valid tool calls after {max_retries} retries.",
                trajectory=messages,
                metadata={"errors": validation_errors},
            )

        # 4. Process tool calls
        # Step 1: Pre-process predicates to fix missing output tokens
        for name, args in parsed_calls:
            if name == "add_predicate":
                content = args.get("content", "")
                outputs = self._normalize_output_ids(args.get("outputs", []))
                # Find <|-E(\d+)-|> or <|-Ei-|> tokens
                tokens = re.findall(r"<\|-([^|]+)-\|>", content)
                for token in tokens:
                    if token not in outputs:
                        outputs.append(token)
                args["outputs"] = outputs

        # Step 2: Collect candidate entities and analyze usage
        candidate_entities: Dict[str, Dict[str, Any]] = {}
        for name, args in parsed_calls:
            if name == "add_entity" and "id" in args:
                candidate_entities[args["id"]] = args

        entity_ids_used_in_outputs = set()
        for name, args in parsed_calls:
            if name == "add_predicate":
                entity_ids_used_in_outputs.update(args.get("outputs", []))

        # Step 3: Filter and add entities
        new_entities: List[str] = []
        all_entity_ids = set(cokg.entities.keys())
        
        for eid, args in candidate_entities.items():
            if eid in cokg.entities:
                continue
            if eid not in entity_ids_used_in_outputs:
                if self.logger:
                    self.logger.log_agent_result("PredicateSampler", f"Entity {eid} not used in any predicate output, skipping", False)
                continue
            
            try:
                entity_data = self._create_entity_from_args(args)
                cokg.add_entity(**entity_data)
                new_entities.append(eid)
                all_entity_ids.add(eid)
                if self.logger:
                    self.logger.log_sampler_entity_added(
                        entity_id=eid,
                        name=entity_data["name"],
                        category=entity_data.get("category", "Unknown"),
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log_agent_result("PredicateSampler", f"Error adding entity {eid}: {e}", False)

        # Step 4: Filter and collect valid predicates
        valid_candidate_predicates: List[Dict[str, Any]] = []
        for name, args in parsed_calls:
            if name != "add_predicate":
                continue
                
            try:
                predicate_data = self._create_predicate_from_args(args)
                input_id = predicate_data.get("input")
                output_ids = predicate_data.get("outputs", [])

                # Check if all output entities exist
                missing = [eid for eid in output_ids if eid not in all_entity_ids]
                if missing:
                    if self.logger:
                        self.logger.log_agent_result("PredicateSampler", f"Skipping predicate: missing output entities {missing}", False)
                    continue
                
                # Check if input entity exists
                if input_id not in all_entity_ids:
                    if self.logger:
                        self.logger.log_agent_result("PredicateSampler", f"Skipping predicate: missing input entity {input_id}", False)
                    continue

                valid_candidate_predicates.append(predicate_data)
            except Exception as e:
                if self.logger:
                    self.logger.log_agent_result("PredicateSampler", f"Error validating predicate candidate: {e}", False)

        # Step 5: Sample if necessary and add to COKG
        if len(valid_candidate_predicates) > 3:
            if self.logger:
                self.logger.log_agent_result("PredicateSampler", f"Sampling 3 predicates from {len(valid_candidate_predicates)} valid candidates", True)
            sampled_predicates_data = random.sample(valid_candidate_predicates, 3)
        else:
            sampled_predicates_data = valid_candidate_predicates

        sampled_predicates: List[Dict[str, Any]] = []
        for predicate_data in sampled_predicates_data:
            cokg.add_predicate(**predicate_data)
            sampled_predicates.append(predicate_data)
            if self.logger:
                self.logger.log_sampler_predicate_added(predicate_data["content"])

        return AgentRunResult(
            status="success",
            answer=f"Processed tool calls, added {len(new_entities)} entities and {len(sampled_predicates)} predicates",
            trajectory=[{"role": "user", "content": user_prompt}, {"role": "assistant", "content": content}],
            metadata={
                "sampled_predicates": sampled_predicates,
                "new_entities": new_entities,
                "tool_calls": len(parsed_calls),
            },
        )

    def _clean_links(self, links: Any) -> List[str]:
        """Resolve link references and ensure uniqueness."""
        if not links:
            return []
            
        if isinstance(links, str):
            try:
                links = json.loads(links)
            except (json.JSONDecodeError, TypeError):
                links = [links] if links else []
        
        if not isinstance(links, list):
            links = [links]

        resolved_links = []
        for link in links:
            actual_url = None
            if isinstance(link, int):
                ref_num = link
                for url, num in self.link_mapping.items():
                    if num == ref_num:
                        actual_url = url
                        break
            elif isinstance(link, str) and link.startswith('[') and link.endswith(']'):
                try:
                    ref_num = int(link[1:-1])
                    for url, num in self.link_mapping.items():
                        if num == ref_num:
                            actual_url = url
                            break
                except (ValueError, IndexError):
                    actual_url = link
            else:
                actual_url = str(link)
            
            if actual_url and actual_url not in resolved_links:
                resolved_links.append(actual_url)
        
        return resolved_links

    def _create_entity_from_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create entity parameters from tool call arguments."""
        return {
            "id": args.get("id", f"E{uuid.uuid4().hex[:4]}"),
            "name": args.get("name", "Unknown Entity"),
            "description": args.get("description", "An entity"),
            "category": args.get("category", "Unknown"),
            "is_variable": False,
            "links": self._clean_links(args.get("links", []))
        }

    def _create_predicate_from_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create predicate parameters from tool call arguments."""
        return {
            "content": args.get("content", ""),
            "input": args.get("input", ""),
            "outputs": self._normalize_output_ids(args.get("outputs", [])),
            "links": self._clean_links(args.get("links", []))
        }

    def _get_next_entity_id(self, existing_entity_ids: List[str]) -> int:
        """Calculate the next available entity ID number."""
        if not existing_entity_ids:
            return 0

        # Extract numeric parts from entity IDs that start with 'E'
        entity_numbers = []
        for entity_id in existing_entity_ids:
            if entity_id.startswith('E'):
                try:
                    # Extract the number after 'E'
                    number = int(entity_id[1:])
                    entity_numbers.append(number)
                except ValueError:
                    continue

        # Return the next available number
        if entity_numbers:
            return max(entity_numbers) + 1
        else:
            return 0
