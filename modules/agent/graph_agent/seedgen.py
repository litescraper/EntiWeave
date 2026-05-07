import os
import json
import random
import string
import time
import asyncio
from typing import List, Dict, Any, Optional

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, BrowserConfig, AgentRunResult
from modules.agent.browser.simple_browser import SimpleBrowser
from modules.agent.memory.base_memory import BaseMemory
from modules.graph.knowledge_graph.cokg import CompositionalKnowledgeGraph, Entity
from modules.logger import BaseAgentLogger
from .utils import format_tools_description

# Topic to valid categories mapping
TOPIC_TO_CATEGORIES = {
    # Academic & Scientific & Technology & Engineering
    "physics": ["person", "scientific_paper", "news_article"],
    "chemistry": ["person", "scientific_paper", "news_article"],
    "biology": ["person", "scientific_paper", "news_article"],
    "mathematics": ["person", "institution", "scientific_paper", "news_article"],
    "computer_science": ["person", "institution", "scientific_paper", "news_article", "code_repo"],
    "medicine": ["person", "institution", "scientific_paper", "news_article", "drug_trial", "modern_drug"],
    "astronomy": ["person", "institution", "scientific_paper", "news_article", "astronomical_object"],
    "artificial_intelligence": ["person", "event", "institution", "scientific_paper", "blog_post", "news_article"],

    # Arts & Culture, Sports & Recreation
    "music": ["person", "event", "company", "music_work", "blog_post", "news_article"],
    "visual_arts": ["person", "event", "institution", "art_work", "place", "blog_post", "news_article"],
    "film": ["person", "event", "institution", "company", "film_work", "blog_post", "news_article"],
    "dance": ["person", "event", "institution", "company", "dance_work", "blog_post", "news_article"],
    "food": ["person", "event", "institution", "company", "food", "product", "blog_post", "news_article"],
    "sports": ["person", "event", "institution", "place", "blog_post", "news_article", "sport_match"],
    "gaming": ["person", "event", "company", "blog_post", "news_article"],

    # Business & Law & Humanities & Social Sciences
    "history": ["person", "event", "institution", "company", "artifact", "place", "blog_post", "news_article", "scientific_paper"],
    "philosophy": ["person", "institution", "scientific_paper", "blog_post", "news_article"],
    "literature": ["person", "institution", "works", "blog_post", "news_article"],
    "psychology": ["person", "institution", "scientific_paper", "blog_post", "news_article"],
    "economics": ["person", "institution", "scientific_paper", "blog_post", "news_article", "economic_report"],
    "corporate": ["person", "event", "company", "blog_post", "news_article", "financial_report", "stock_market", "merger_acquisition", "scandal"],
    "government": ["person", "institution", "blog_post", "news_article", "financial_report", "policy", "enacted_law", "government_survey", "international_report"],
    "law": ["person", "event", "institution", "blog_post", "news_article", "law_document", "legal_case"],
    "entrepreneurship": ["person", "event", "company", "blog_post", "news_article", "financial_report", "stock_market"],
    "marketing": ["person", "company", "blog_post", "news_article", "marketing_campaign"],

    # Spatial Information
    # "location": ["restaurant", "hotel", "cafe", "hospital", "school", "park", "museum", "subway_line", "bus_line", "road"]
}

SCIENCE_TOPICS = ["physics", "chemistry", "biology", "mathematics", "computer_science", "medicine", "astronomy", "artificial_intelligence"]
ART_TOPICS = ["music", "visual_arts", "film", "dance", "food", "sports", "gaming"]
HUMANITIES_TOPICS = ["history", "philosophy", "literature", "psychology", "economics", "corporate", "government", "law", "entrepreneurship", "marketing"]
SPATIAL_TOPICS = ["location"]

VALID_TOPICS = list(TOPIC_TO_CATEGORIES.keys())

# Context randomization for diversity
COUNTRIES = [
    "United States", "United Kingdom", "China", "Japan", "Germany", 
    "France", "India", "Brazil", "Russia", "South Korea", 
    "Canada", "Australia", "Italy", "Spain", "Mexico", 
    "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", 
    "Singapore" ,"Malaysia", "South Africa", "Nigeria", "Kenya"
]

YEARS = [str(year) for year in range(1980, 2026)]

class SeedGenAgent(BaseLLMAgent):
    """
    An agent that finds an unpopular/obscure entity and its description.
    It handles both standard entities and internet objects (blog posts, papers, etc.).
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[BaseAgentLogger] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the SeedGenAgent.
        """
        # Load prompts and tools from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        prompt_data = self._load_prompts(yaml_path)

        # Only the research phase should receive tools. The proposal phase is a
        # plain generation step and should not be allowed to emit tool calls.
        self.research_tools = tools if tools is not None else prompt_data.get("seedgen_tools", [])
        self.effective_tools = self.research_tools
        
        # Load specific prompts
        self.propose_system_prompt = prompt_data.get("seedgen_propose_system_prompt", "")
        self.propose_user_prompt = prompt_data.get("seedgen_propose_user_prompt", "")
        self.research_system_prompt = prompt_data.get("seedgen_research_system_prompt", "")
        
        super().__init__(
            config=config,
            system_prompt="", # Will be set per run
            tools=[]
        )

        self.browser = SimpleBrowser(config=browser_config)
        self.logger = logger
        self.random = random.Random(seed) if seed is not None else random.Random()

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Execute search tools or finish.
        """
        if tool_name == "finish":
            return f"Seed entity found: {tool_args.get('entity_name')}"
        
        # Browser tools
        try:
            response = self.browser.do(tool_name, **tool_args)
            return self.browser.format_response_as_text(response)
        except Exception as e:
            print(e)
            return f"Error executing {tool_name}: {str(e)}"

    def _extract_tags(self, text: str) -> Dict[str, str]:
        """
        Extracts content from <entity> or <seed_query> tags.
        """
        import re
        results = {}
        entity_match = re.search(r"<entity>(.*?)</entity>", text, re.DOTALL)
        if entity_match:
            results["entity"] = entity_match.group(1).strip()
        
        query_match = re.search(r"<seed_query>(.*?)</seed_query>", text, re.DOTALL)
        if query_match:
            results["seed_query"] = query_match.group(1).strip()
            
        return results

    def _select_topic_category(self, topic: Optional[str], category: Optional[str]) -> tuple[str, str]:
        """Selects topic and category, using random choice if not provided."""
        if not topic and not category:
            all_pairs = []
            for t in SCIENCE_TOPICS + ART_TOPICS + HUMANITIES_TOPICS + SPATIAL_TOPICS:
                for c in TOPIC_TO_CATEGORIES.get(t, []):
                    all_pairs.append((t, c))
            return self.random.choice(all_pairs)
        
        selected_topic = topic or self.random.choice(VALID_TOPICS)
        if category:
            if category not in TOPIC_TO_CATEGORIES.get(selected_topic, []):
                raise ValueError(f"Category '{category}' is not valid for topic '{selected_topic}'. Valid categories: {TOPIC_TO_CATEGORIES.get(selected_topic, [])}")
            selected_category = category
        else:
            valid_categories = TOPIC_TO_CATEGORIES.get(selected_topic, [])
            selected_category = self.random.choice(valid_categories)
        
        return selected_topic, selected_category

    async def _propose_target(
        self, 
        topic: str, 
        category: str, 
        revision_context: str, 
        extra_context: str
    ) -> tuple[Optional[str], Optional[str], int]:
        """Phase 1: Propose a target (Entity or Seed Query)."""
        propose_user_msg = self.propose_user_prompt.format(
            topic=topic,
            category=category,
            revision_context=revision_context + extra_context
        )
        
        propose_messages = [{"role": "user", "content": propose_user_msg}]
        proposed_entity = None
        proposed_query = None
        turns = 0
        
        max_proposal_retries = 3
        for proposal_attempt in range(max_proposal_retries):
            formatted_propose_messages = BaseMemory.format_messages(
                propose_messages, 
                sampler_type=self.config.sampler_type, 
                include_thinking=True
            )
            
            proposal_response = self.generate_response(
                formatted_propose_messages,
                system=self.propose_system_prompt,
                include_system=True
            )
            turns += 1
            
            tags = self._extract_tags(proposal_response.content)
            proposed_entity = tags.get("entity")
            proposed_query = tags.get("seed_query")
            
            if proposed_entity or proposed_query:
                break
            
            if self.logger:
                self.logger.log_agent_action(
                    "SeedGen", 
                    f"Proposal attempt {proposal_attempt + 1} failed", 
                    "No <entity> or <seed_query> tags found. Retrying..."
                )
            
            propose_messages.append({
                "role": "assistant", 
                "content": proposal_response.content,
                "reasoning_content": proposal_response.reasoning_content
            })
            propose_messages.append({
                "role": "user",
                "content": "You forgot to wrap your proposal in <entity> or <seed_query> tags. Please provide the proposal again with the correct tags."
            })

        # Fallback if no tags found after retries
        if not proposed_entity and not proposed_query:
            proposed_entity = proposal_response.content.strip().strip('"').strip("'")
            
        return proposed_entity, proposed_query, turns

    async def _research_target(
        self,
        topic: str,
        category: str,
        proposed_entity: Optional[str],
        proposed_query: Optional[str],
        country_year_context: str,
        max_steps: int
    ) -> tuple[Optional[Entity], List[Dict[str, Any]], bool, Optional[str]]:
        """Phase 2: Collect information and grounding links for the proposed target."""
        if proposed_entity:
            research_target = f"entity '{proposed_entity}'"
            start_msg = f"The entity to research is: '{proposed_entity}'."
        elif proposed_query:
            research_target = f"internet object using query '{proposed_query}'"
            start_msg = f"Find a specific, obscure internet object (blog post, article, or paper) related to the query: '{proposed_query}'."
        else:
            # Should not happen with fallback in _propose_target
            return None, [], False, None

        if self.logger:
            self.logger.log_agent_action(
                "SeedGen", 
                "Researching", 
                f"[bold cyan]{research_target}[/]"
            )

        tools_description = format_tools_description(self.effective_tools)
        effective_research_system_prompt = self.research_system_prompt.format(
            topic=topic,
            category=category,
            tools_description=tools_description,
            extra_context=country_year_context
        )

        memory = BaseMemory(system_prompt=effective_research_system_prompt)
        memory.add_user_message(
            f"{start_msg}\n\n"
            f"Please perform a COMPREHENSIVE search to gather detailed context, description, and MUST find grounding links. "
            f"For internet objects, the link must be a direct URL to the object itself.\n\n"
            f"When you have enough information, use the 'finish' tool to provide its details (name, category, content, links, reasoning)."
        )
        
        should_revise = False
        revision_reason = None

        for step in range(max_steps):
            if step == max_steps - 1:
                memory.add_user_message("Observation: This is the FINAL step. You must output {'action': 'finish', ...} now. Do not search again.")
            
            response = self.sampler(
                memory.get_formatted_messages(sampler_type=self.config.sampler_type, include_thinking=True),
                tools=self.research_tools
            )
            
            tool_calls = response.tool_calls or []
            memory.add_assistant_message(
                content=response.content,
                reasoning_content=response.reasoning_content,
                tool_calls=tool_calls
            )
            
            if not tool_calls:
                memory.add_user_message("Observation: No valid tool call found. Please use search tools to find info or 'finish' to conclude.")
                continue

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

                
                if self.logger:
                    self.logger.log_agent_tool_execution("SeedGen", tool_name, tool_args)

                if tool_name == "finish":
                    entity_name = tool_args.get("entity_name", proposed_entity or "Unknown")
                    entity_content = tool_args.get("content", "")
                    entity_category = tool_args.get("category", category)
                    source_indices = tool_args.get("links", [])
                    
                    entity_links = []
                    if isinstance(source_indices, str):
                        try:
                            idx_list = json.loads(source_indices)
                        except (json.JSONDecodeError, TypeError):
                            if source_indices.startswith("http://") or source_indices.startswith("https://"):
                                entity_links.append(source_indices)
                                idx_list = []
                            else:
                                memory.add_user_message(f"Error: Invalid source indices: {source_indices}")
                                continue
                    else:
                        idx_list = source_indices

                    for idx in idx_list:
                        try:
                            idx_int = int(idx)
                            if 0 <= idx_int < len(self.browser.references):
                                entity_links.append(self.browser.references[idx_int].url)
                        except (ValueError, TypeError):
                            pass

                    entity = Entity(
                        id="E0",
                        name=entity_name,
                        description=entity_content,
                        category=entity_category,
                        is_variable=False,
                        links=entity_links
                    )
                    return entity, memory.get_messages(), False, None

                if tool_name == "revise":
                    revision_reason = tool_args.get("reason", "No reason provided.")
                    if self.logger:
                        self.logger.log_agent_result("SeedGen", f"Revision requested: {revision_reason}", False)
                    return None, memory.get_messages(), True, revision_reason

                observation = await self.call_tool(tool_name, tool_args)
                memory.add_tool_message(tool_call_id, f"Observation: {observation}", tool_name=tool_name)
        
        return None, memory.get_messages(), False, None

    async def run(
        self, 
        topic: Optional[str] = None, 
        category: Optional[str] = None, 
        max_steps: int = 10,
        max_revisions: int = 3,
        seed: Optional[int] = None
    ) -> AgentRunResult:
        """Runs the agent to generate a seed entity."""
        if seed is not None:
            self.random = random.Random(seed)
            
        selected_topic, selected_category = self._select_topic_category(topic, category)
        
        if self.logger:
            self.logger.log_agent_action(
                "SeedGen", 
                f"Topic: [bold magenta]{selected_topic}[/]", 
                f"Generating seed for category: [bold green]{selected_category}[/]"
            )
        
        revision_details = []
        revision_context = ""
        
        # Determine if we should add country and year context
        target_categories = []
        target_categories = ["financial_report", "legal_case", "film_work", "music_work", "stock_market", "government_survey", "enacted_law", "policy"]
        extra_context = ""
        country_year_context = ""
        
        # Location only needs country; corporate and target categories need both
        if selected_topic == "location":
            random_country = self.random.choice(COUNTRIES)
            extra_context = f"\nAdditional Context: Focus on targets from {random_country}."
            country_year_context = f"Country: {random_country}"
        elif selected_topic == "corporate" or selected_category in target_categories:
            random_country = self.random.choice(COUNTRIES)
            random_year = self.random.choice(YEARS)
            extra_context = f"\nAdditional Context: Focus on targets from {random_country} in the year {random_year}."
            country_year_context = f"Country: {random_country}\nYear: {random_year}"

        final_memory_trajectory = []
        
        for revision_attempt in range(max_revisions):
            rev_start_time = time.time()
            
            if self.logger:
                self.logger.log_agent_action(
                    "SeedGen", 
                    f"Attempt [bold yellow]{revision_attempt + 1}/{max_revisions}[/]", 
                    "Proposing and researching target..."
                )

            # --- Phase 1: Propose ---
            proposed_entity, proposed_query, prop_turns = await self._propose_target(
                selected_topic, selected_category, revision_context, extra_context
            )
            
            # --- Phase 2: Research ---
            entity, trajectory, should_revise, revision_reason = await self._research_target(
                selected_topic, 
                selected_category, 
                proposed_entity, 
                proposed_query, 
                country_year_context, 
                max_steps
            )
            
            final_memory_trajectory = trajectory
            
            if entity:
                revision_details.append({
                    "attempt": revision_attempt + 1,
                    "time": time.time() - rev_start_time
                })
                self.browser.reset()
                return AgentRunResult(
                    answer=json.dumps(entity.to_dict(), indent=2, ensure_ascii=False),
                    trajectory=trajectory,
                    status="success",
                    turns=revision_attempt + 1,
                    metadata={
                        "entity": entity.to_dict(),
                        "topic": selected_topic,
                        "category": selected_category,
                        "revisions": revision_attempt + 1,
                        "revision_details": revision_details
                    }
                )

            if should_revise:
                research_target = proposed_entity or proposed_query
                revision_context = f"\nPrevious attempt for '{research_target}' failed because: {revision_reason}\nPlease propose a DIFFERENT and BETTER target."
                revision_details.append({
                    "attempt": revision_attempt + 1,
                    "time": time.time() - rev_start_time,
                    "reason": revision_reason
                })
                continue
            else:
                # If we finished max_steps without success or revision request
                break

        return AgentRunResult(
            answer="Max attempts reached without finding a seed entity.",
            trajectory=final_memory_trajectory,
            status="max_revisions_reached",
            metadata={
                "revisions": revision_attempt + 1,
                "revision_details": revision_details
            }
        )
