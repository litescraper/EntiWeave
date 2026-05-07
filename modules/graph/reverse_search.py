"""
Reverse Search Logic

This module implements the core reverse search algorithm for expanding a knowledge graph.
It can be used to start a new search or continue from an existing state.
"""

import os
import re
import random
from typing import Dict, List, Optional, Any, Set, Tuple

from modules.graph.knowledge_graph.cokg import (
    CompositionalKnowledgeGraph,
    Entity,
    Predicate,
)
from modules.agent.graph_agent.knowledge_search import KnowledgeSearchAgent
from modules.agent.graph_agent.predicate_sampler import PredicateSamplerAgent
from modules.agent.graph_agent.fuzzifier import Fuzzifier
from modules.agent.graph_agent.graph_evolver import GraphEvolver
from modules.agent.graph_agent.fact_checker import FactCheckerAgent
from modules.agent.agent_types import AgentSamplerConfig, BrowserConfig, AgentRunResult
from modules.graph.database import DatabaseManager
from modules.logger import TerminalLogger


def format_predicate_with_entity_names(
    predicate: Predicate, cokg: CompositionalKnowledgeGraph
) -> Dict[str, Any]:
    """
    Replace entity ID placeholders in predicate content with actual entity names.

    Args:
        predicate: The predicate object to format
        cokg: The knowledge graph containing entities

    Returns:
        Dictionary with formatted predicate content
    """
    content = predicate.content

    # Replace all entity placeholders with their names
    for entity_id, entity in cokg.entities.items():
        placeholder = f"<|-{entity_id}-|>"
        if placeholder in content:
            content = content.replace(placeholder, f'"{entity.name}"')

    return {
        "id": predicate.id,
        "content": content,
        "input": predicate.input,
        "outputs": predicate.outputs,
        "links": predicate.links,
        "topic": getattr(predicate, "topic", "General"),  # Add topic if available
    }


def collect_ancestor_predicates(
    entity_id: str,
    entity_parent_context: Dict[str, Any],
    cokg: CompositionalKnowledgeGraph,
) -> List[Dict[str, Any]]:
    """
    Walk up the ancestry tree from the given entity to the root and collect all predicates
    involving ancestor entities, with entity masks replaced by actual entity names.

    Args:
        entity_id: The entity ID to start walking from
        entity_parent_context: The context mapping with ancestry paths
        cokg: The knowledge graph containing entities and predicates

    Returns:
        List of all predicates involving ancestor entities with formatted content
    """
    ancestor_predicates = []

    # Get the ancestry path for this entity
    ancestry_path = entity_parent_context.get(entity_id, {}).get("ancestry_path", [])

    # Walk through each ancestor in the path
    for ancestor_id in ancestry_path:
        if ancestor_id in cokg.entities:
            # Find all predicates where this ancestor is involved (as input or output)
            for predicate_id, predicate in cokg.predicates.items():
                if predicate.input == ancestor_id or ancestor_id in predicate.outputs:
                    # Format the predicate with entity names and avoid duplicates
                    formatted_predicate = format_predicate_with_entity_names(
                        predicate, cokg
                    )
                    if formatted_predicate not in ancestor_predicates:
                        ancestor_predicates.append(formatted_predicate)

    return ancestor_predicates


async def run_reverse_search(
    cokg: CompositionalKnowledgeGraph,
    run_id: str,
    graph_id: str,
    queue: List[str],
    processed_entities: Set[str],
    entity_parent_context: Dict[str, Any],
    search_agent: KnowledgeSearchAgent,
    sampler_agent: PredicateSamplerAgent,
    max_complexity_budget: int = 10,
    max_search_steps: int = 5,
    prune_factor: float = 0.9,
    db_manager: Optional[DatabaseManager] = None,
    logger: Optional[TerminalLogger] = None,
    seed_object_id: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> CompositionalKnowledgeGraph:
    """
    Execute the reverse search process.

    Args:
        cokg: The initial knowledge graph
        run_id: The run identifier for this search
        graph_id: The graph identifier for this search
        queue: List of entity IDs to expand
        processed_entities: Set of entity IDs already processed
        entity_parent_context: Context mapping for informed sampling
        search_agent: Knowledge search agent
        sampler_agent: Predicate sampler agent
        max_complexity_budget: Maximum number of entities to expand to
        max_search_steps: Maximum search steps per entity
        prune_factor: Geometric decay factor for expansion probability (default 0.95)
        db_manager: Optional database manager for saving intermediate results
        logger: Optional terminal logger
        seed_object_id: Optional seed object ID for checkpointing
        stats: Optional dictionary to collect statistics
    """
    # 2. Main expansion loop
    try:
        current_complexity = len(cokg.entities)
        initial_complexity = current_complexity
        expanded_count_this_run = 0

        if stats is not None and "graph_expansion" not in stats:
            stats["graph_expansion"] = []

        while queue and current_complexity < max_complexity_budget:
            # Pop next entity from queue
            current_entity_id = queue.pop(0)
            if current_entity_id in processed_entities:
                continue

            current_entity = cokg.entities[current_entity_id]
            current_entity.is_variable = True  # Mark as variable when expansion begins

            # Track expansion details for this entity
            expansion_detail = {
                "entity_id": current_entity_id,
                "entity_name": current_entity.name,
                "knowledge_search": {},
                "predicate_sampling": {},
            }

            # Pruning mechanism: probability of expansion decreases geometrically
            expansion_prob = prune_factor**expanded_count_this_run
            if random.random() > expansion_prob:
                if logger:
                    logger.warning(
                        f"Skipping expansion of {current_entity.name} ({current_entity_id}) - Probability: {expansion_prob:.4f}"
                    )
                processed_entities.add(current_entity_id)
                continue

            if logger:
                logger.log_entity_expansion_start(
                    current_entity.name,
                    current_entity_id,
                    expansion_prob,
                    current_complexity,
                    max_complexity_budget,
                )

            # Knowledge search phase - discover predicates
            search_results = await search_agent.run(
                entity=current_entity, max_steps=max_search_steps, run_id=run_id
            )

            # Record search statistics in detail
            expansion_detail["knowledge_search"] = {
                "turns": search_results.turns,
                "time": search_results.total_time,
                "input_tokens": search_results.input_tokens,
                "output_tokens": search_results.output_tokens,
                "status": search_results.status,
            }

            if search_results.status not in [
                "success",
                "max_steps_reached",
                "completed",
            ]:
                print(f"Search failed for entity {current_entity_id}, skipping...")
                if stats is not None:
                    stats["graph_expansion"].append(expansion_detail)
                processed_entities.add(current_entity_id)
                continue

            discovered_predicates = search_results.metadata.get("predicates", [])

            # Predicate sampling and entity expansion phase
            if discovered_predicates:
                if logger:
                    # Display discovered predicates table with discovery information
                    logger.display_predicates_table_detailed(
                        discovered_predicates,
                        "Discovered Predicates",
                        entity_name=current_entity.name,
                    )

                # Get parent context for informed sampling - now includes all ancestor predicates
                ancestor_predicates = collect_ancestor_predicates(
                    current_entity_id, entity_parent_context, cokg
                )
                parent_context = {
                    "ancestor_predicates": ancestor_predicates,
                    "ancestry_path": entity_parent_context.get(
                        current_entity_id, {}
                    ).get("ancestry_path", []),
                }

                if logger:
                    logger.log_predicate_sampling_start(
                        current_entity.name, len(discovered_predicates)
                    )

                # Collect existing entities information for context
                existing_entities = []
                for entity_id, entity in cokg.entities.items():
                    existing_entities.append(
                        {
                            "id": entity_id,
                            "name": entity.name,
                            "description": entity.description,
                        }
                    )

                sampling_result = sampler_agent.run(
                    current_entity,
                    discovered_predicates,
                    cokg,
                    parent_context=parent_context,
                    existing_entities=existing_entities,
                )

                # Record sampling statistics
                expansion_detail["predicate_sampling"] = {
                    "time": sampling_result.total_time,
                    "input_tokens": sampling_result.input_tokens,
                    "output_tokens": sampling_result.output_tokens,
                    "status": sampling_result.status,
                }

                if logger:
                    logger.log_sampling_complete(
                        entity_name=current_entity.name,
                        entities_added=len(
                            sampling_result.metadata.get("new_entities", [])
                        ),
                        predicates_added=len(
                            sampling_result.metadata.get("sampled_predicates", [])
                        ),
                        message=sampling_result.answer,
                    )

                # Add newly discovered entities to queue
                for new_entity_id in sampling_result.metadata.get("new_entities", []):
                    if (
                        new_entity_id not in processed_entities
                        and new_entity_id not in queue
                    ):
                        queue.append(new_entity_id)
                        # Build ancestry path: current entity's path + current entity
                        current_ancestry = entity_parent_context.get(
                            current_entity_id, {}
                        ).get("ancestry_path", [])
                        entity_parent_context[new_entity_id] = {
                            "ancestry_path": current_ancestry + [current_entity_id],
                            "ancestor_predicates": current_ancestry
                            + [
                                current_entity_id
                            ],  # Will be populated by tree walking function
                        }
                        new_entity = cokg.entities[new_entity_id]

            # Add this entity's expansion details to stats
            if stats is not None:
                stats["graph_expansion"].append(expansion_detail)

            # Update tracking and check termination conditions
            processed_entities.add(current_entity_id)
            expanded_count_this_run += 1
            current_complexity = len(cokg.entities)

            if db_manager and seed_object_id:
                try:
                    await db_manager.save_generated_graph(
                        cokg=cokg,
                        run_id=run_id,
                        graph_id=graph_id,
                        seed_object_id=seed_object_id,
                        new_entities=queue,
                        graph_type="original",
                        stats={"graph_expansion": stats.get("graph_expansion", [])}
                        if stats
                        else {},
                    )
                    if logger:
                        logger.info(
                            f"Checkpoint saved for Graph {graph_id} after expanding {current_entity_id}"
                        )
                except Exception as e:
                    if logger:
                        logger.error(f"Error saving checkpoint: {e}")
                    else:
                        print(f"Error saving checkpoint: {e}")

            if current_complexity >= max_complexity_budget:
                print(
                    f"Reached complexity budget ({max_complexity_budget}). Stopping expansion."
                )
                break
            else:
                # randomize queue position
                random.shuffle(queue)

        # Final summary
        if logger:
            logger.info(f"\n--- Reverse Search Phase Complete ---")
            logger.info(f"Started with: {initial_complexity} entities")
            logger.info(f"Ended with: {current_complexity} entities")
            logger.info(f"Processed: {len(processed_entities)} entities")
            logger.info(f"Remaining in queue: {len(queue)} entities")

        return cokg

    except Exception as e:
        print(f"Error in reverse search: {e}")
        raise


async def evolve_graph(
    cokg: CompositionalKnowledgeGraph,
    evolver_agent: GraphEvolver,
    fuzzifier_agent: Fuzzifier,
    fuzz_constant_prob: float = 0.7,
    fuzz_use_category_prob: float = 0.5,
    run_id: Optional[str] = None,
    graph_id: Optional[str] = None,
    db_manager: Optional[DatabaseManager] = None,
    stats: Optional[Dict[str, Any]] = None,
    logger: Optional[TerminalLogger] = None,
) -> CompositionalKnowledgeGraph:
    """
    Evolve the graph in two steps:
    1. Evolve the predicates and constants using GraphEvolver (pruning/evolving).
    2. Fuzzify the remaining constant entities using Fuzzifier.

    Args:
        cokg: The knowledge graph to evolve
        evolver_agent: The graph evolver agent
        fuzzifier_agent: The fuzzifier agent to use for generating fuzzy descriptions
        fuzz_constant_prob: Probability of fuzzing constant entities (0.0 to 1.0, default: 0.7)
        fuzz_use_category_prob: Probability of using category as fuzzy name instead of agent (0.0 to 1.0, default: 0.5)
        run_id: Optional run identifier
        graph_id: Optional graph identifier
        db_manager: Optional database manager for saving statistics
        stats: Optional dictionary to collect statistics
        logger: Optional terminal logger
    """
    if stats is not None:
        if "graph_evolution" not in stats:
            stats["graph_evolution"] = []
        if "fuzzification" not in stats:
            stats["fuzzification"] = []

    # 1. Evolve the graph (pruning/evolving predicates and constants)
    if logger:
        logger.info("\n--- Step 1: Evolving Graph Predicates and Constants ---")

    try:
        evolver_result = await evolver_agent.run(cokg)
        if stats is not None:
            stats["graph_evolution"].append(
                {
                    "status": evolver_result.status,
                    "answer": evolver_result.answer,
                    "time": evolver_result.total_time,
                    "input_tokens": evolver_result.input_tokens,
                    "output_tokens": evolver_result.output_tokens,
                }
            )
        if logger:
            logger.info(f"Graph evolution complete. Status: {evolver_result.status}")
    except Exception as e:
        print(f"Error during graph evolution: {e}")
        if logger:
            logger.error(f"Error during graph evolution: {e}")

    # 2. Fuzzify remaining entities
    if logger:
        logger.info("\n--- Step 2: Fuzzifying Remaining Entities ---")

    for entity_id, entity in cokg.entities.items():
        if entity_id == "E0":
            if logger:
                logger.info(f"Skipping fuzzification for root entity E0: {entity.name}")
            continue

        original_name = entity.name

        if not entity.is_variable:
            # Constant entity - decide whether to fuzzify based on probability
            if random.random() < fuzz_constant_prob:
                # Decide whether to use category directly or call fuzzifier agent
                if random.random() < fuzz_use_category_prob:
                    if logger:
                        logger.info(
                            f"Using category as fuzzy name for constant entity: {original_name} ({entity_id}) -> {entity.category}"
                        )
                    entity.name = entity.category
                    if stats is not None:
                        stats["fuzzification"].append(
                            {
                                "entity_id": entity_id,
                                "entity_name": original_name,
                                "type": "constant",
                                "method": "category",
                                "status": "success",
                            }
                        )
                else:
                    if logger:
                        logger.info(
                            f"Fuzzifying constant entity with agent: {original_name} ({entity_id})..."
                        )
                    try:
                        fuzz_result = await fuzzifier_agent.run(entity)

                        # Record fuzzification statistics
                        if stats is not None:
                            stats["fuzzification"].append(
                                {
                                    "entity_id": entity_id,
                                    "entity_name": original_name,
                                    "type": "constant",
                                    "method": "agent",
                                    "turns": fuzz_result.turns,
                                    "time": fuzz_result.total_time,
                                    "input_tokens": fuzz_result.input_tokens,
                                    "output_tokens": fuzz_result.output_tokens,
                                    "status": fuzz_result.status,
                                }
                            )

                        if fuzz_result.status == "success":
                            entity.name = fuzz_result.answer
                            # Add supporting links to entity if provided
                            if (
                                "links" in fuzz_result.metadata
                                and fuzz_result.metadata["links"]
                            ):
                                entity.links = list(
                                    set(entity.links + fuzz_result.metadata["links"])
                                )
                        else:
                            if logger:
                                logger.warning(
                                    f"Warning: Fuzzifier failed for {original_name}, using original name."
                                )
                    except Exception as e:
                        if logger:
                            logger.error(f"Error fuzzifying {original_name}: {e}")

                # Update description
                entity.description = f"[{original_name}] {entity.description}"
            else:
                if logger:
                    logger.info(
                        f"Skipping fuzzification for constant entity: {original_name} ({entity_id}) (probability: {fuzz_constant_prob})"
                    )
        else:
            # Variable entity
            if logger:
                logger.info(
                    f"Fuzzifying variable entity: {original_name} ({entity_id}) -> {entity.category}"
                )
            entity.name = entity.category
            entity.description = f"[{original_name}] {entity.description}"

    return cokg


async def fact_check_graph(
    cokg: CompositionalKnowledgeGraph,
    fact_checker_agent: FactCheckerAgent,
    logger: Optional[TerminalLogger] = None,
) -> Dict[str, Any]:
    """
    Perform fact checking on the graph entities.

    1. First iterate all constant entity and do reverse fuzzy by fetching entity name in [ ]
       at start of the description, name it name and instead put the name as description.
    2. Fact check E0, variable entities, and constant entities.
    """
    results = {}
    constant_fuzzy_descriptions = {}

    # 1. Reverse fuzzy and recover names for all entities (to enable correct predicate parsing)
    for entity_id, entity in cokg.entities.items():
        match = re.match(r"\[(.*?)\]\s*(.*)", entity.description)
        if match:
            real_name = match.group(1)
            desc = match.group(2)

            fuzzy_desc = entity.name

            # Update name for all fuzzed entities
            entity.name = real_name

            if entity.is_variable:
                entity.description = desc
            else:
                entity.description = fuzzy_desc

    # 2. Fact checking
    for entity_id, entity in cokg.entities.items():
        if entity_id == "E0":
            # E0 Special: Check description separately
            if logger:
                logger.info(f"Fact checking E0 description for {entity.name}...")

            description_check = await fact_checker_agent.run(
                entity_name=entity.name, statement=entity.description, urls=entity.links
            )
            results["seed_description"] = {
                "entity_name": entity.name,
                "statement": entity.description,
                "supported": description_check.answer == "True",
                "reasoning": description_check.metadata.get("reasoning", ""),
                "evidence": description_check.metadata.get("evidence", []),
            }

        # Common statement/links gathering for all entities
        statement = None
        links = []

        if entity.is_variable:
            # Variable entities (including E0): gather predicates, parse names, gather links
            predicates = [p for p in cokg.predicates.values() if p.input == entity_id]
            if predicates:
                formatted_contents = []
                all_links = set(entity.links)
                for p in predicates:
                    # Replace entity tokens <|E...|> with actual names
                    formatted = format_predicate_with_entity_names(p, cokg)
                    formatted_contents.append(formatted["content"])
                    all_links.update(p.links)

                # Compilation of the predicates as the statement
                statement = " ".join(formatted_contents)
                links = list(all_links)
            else:
                if logger:
                    logger.info(
                        f"Skipping predicate fact check for variable {entity_id} ({entity.name}): No predicates found."
                    )
        else:
            # Constant entity: cdescription as statement
            statement = constant_fuzzy_descriptions.get(entity_id)
            if not statement:
                # Fallback if no [] was found (e.g., if skipping fuzzification)
                statement = entity.description
            links = entity.links

        if not statement or not links:
            continue

        if logger:
            logger.info(f"Fact checking {entity_id} ({entity.name})...")

        check_result = await fact_checker_agent.run(
            entity_name=entity.name, statement=statement, urls=links
        )

        results[entity_id] = {
            "entity_name": entity.name,
            "statement": statement,
            "supported": check_result.answer == "True",
            "reasoning": check_result.metadata.get("reasoning", ""),
            "evidence": check_result.metadata.get("evidence", []),
        }

    return results
