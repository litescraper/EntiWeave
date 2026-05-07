import os
import yaml
import json
import re
from typing import Any, Dict, List, Optional, Union
from .base_tool import BaseTool
from ..browser.simple_browser import SimpleBrowser
from ..agent_types import BrowserConfig, PageContent

class SearchTool(BaseTool):
    """
    A tool for searching the web.
    """
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "A tool for searching the web"

    async def call(self, **kwargs: Any) -> str:
        browser = kwargs.get("browser")
        if not browser:
            return "Error: No browser provided to search tool."
        
        query = kwargs.get("query")
        if not query:
            return "Error: No query provided to search tool."

        if isinstance(query, (list, tuple)):
            observations = []
            for q in query:
                response = browser.do("search", query=q, **{k: v for k, v in kwargs.items() if k not in ["query", "browser"]})
                observations.append(browser.format_response_as_text(response))
            return "\n\n".join(observations)
        
        # Pass everything except 'browser' to browser.do
        browser_args = {k: v for k, v in kwargs.items() if k != "browser"}
        response = browser.do("search", **browser_args)
        return browser.format_response_as_text(response)

class VisitPageTool(BaseTool):
    """
    A tool for visiting a specific website.
    """
    def __init__(self, name: str = "visit"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Open a specific website. Get content from any website with its URL."

    async def call(self, **kwargs: Any) -> str:
        browser = kwargs.get("browser")
        if not browser:
            return "Error: No browser provided to open tool."
            
        url = kwargs.get("url")
        if not url:
            return "Error: No url provided to open tool."

        if isinstance(url, (list, tuple)):
            observations = []
            for u in url:
                response = browser.do(self.name, url=u, **{k: v for k, v in kwargs.items() if k not in ["url", "browser"]})
                observations.append(browser.format_response_as_text(response))
            return "\n\n".join(observations)
        
        browser_args = {k: v for k, v in kwargs.items() if k != "browser"}
        response = browser.do(self.name, **browser_args)
        return browser.format_response_as_text(response)

class ClickTool(BaseTool):
    """
    A tool for clicking on a link.
    """
    @property
    def name(self) -> str:
        return "click"

    @property
    def description(self) -> str:
        return "A tool for clicking on a link"

    async def call(self, **kwargs: Any) -> str:
        browser = kwargs.get("browser")
        if not browser:
            return "Error: No browser provided to click tool."

        link_id = kwargs.get("link_id")
        if link_id is None:
            return "Error: No link_id provided to click tool."

        try:
            if isinstance(link_id, (list, tuple)):
                observations = []
                for l_id in link_id:
                    processed_id = int(l_id) if isinstance(l_id, str) else l_id
                    new_kwargs = {k: v for k, v in kwargs.items() if k not in ["link_id", "browser"]}
                    response = browser.do("click", link_id=processed_id, **new_kwargs)
                    observations.append(browser.format_response_as_text(response))
                return "\n\n".join(observations)
            
            processed_id = int(link_id) if isinstance(link_id, str) else link_id
            new_kwargs = {k: v for k, v in kwargs.items() if k != "browser"}
            new_kwargs["link_id"] = processed_id
            response = browser.do("click", **new_kwargs)
            return browser.format_response_as_text(response)
        except (ValueError, TypeError):
            return f"Error: link_id must be an integer, got {link_id}"

class RetrieveTool(BaseTool):
    """
    A tool for retrieving and extracting information from a specific website.
    """
    @property
    def name(self) -> str:
        return "retrieve"

    @property
    def description(self) -> str:
        return "A tool for retrieving and extracting information from a specific website."

    async def _extract_info(self, sampler: Any, content: str, query: str) -> str:
        """Helper method to call the sampler for extraction with retry logic"""
        # Load prompts from prompts.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "..", "search_agent", "prompts.yaml")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                prompt_data = yaml.safe_load(f)
        except Exception as e:
            return f"Error loading prompts: {str(e)}"
            
        system_prompt = prompt_data.get("extractor_system_prompt", "")
        user_prompt_template = prompt_data.get("extractor_user_prompt", "")
        
        if not system_prompt or not user_prompt_template:
            return "Error: Extractor prompts not found in prompts.yaml"
            
        user_prompt = user_prompt_template.format(webpage_content=content, goal=query)
        
        # Call sampler with system and user messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        last_raw_content = ""
        for attempt in range(3):
            try:
                response = sampler(messages)
                raw_content = response.reasoning_content + response.content
                last_raw_content = raw_content
                
                summary = self._parse_summary(raw_content)
                if summary is not None:
                    # Capture trajectory ONLY if parsing is successful
                    if "conversations" not in self.metadata:
                        self.metadata["conversations"] = []
                    
                    self.metadata["conversations"].append({
                        "messages": messages,
                        "response": raw_content,
                        "reasoning_content": getattr(response, "reasoning_content", None),
                        "usage": {
                            "input_tokens": getattr(response, "input_tokens", 0),
                            "output_tokens": getattr(response, "output_tokens", 0)
                        }
                    })
                    return summary
            except Exception as e:
                last_raw_content = f"Error during extraction attempt {attempt + 1}: {str(e)}"
                
        # If all retries fail, return the last raw content (without adding to trajectory)
        return last_raw_content

    def _parse_summary(self, text: str) -> Optional[str]:
        """Extracts the 'summary' field from the sampler's JSON response. Returns None if parsing fails."""
        try:
            # 1. Try finding JSON within markdown code blocks
            json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            json_str = json_match.group(1).strip() if json_match else text.strip()
            
            # 2. If no code block or parsing fails, try finding the outermost { }
            if not json_match:
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    json_str = text[start:end+1]

            # 3. Parse JSON and return summary
            data = json.loads(json_str)
            if isinstance(data, dict) and "summary" in data:
                return str(data["summary"])
        except (json.JSONDecodeError, Exception):
            pass
            
        return None

    async def call(self, **kwargs: Any) -> str:
        browser = kwargs.get("browser")
        sampler = kwargs.get("sampler")
        if not browser or not sampler:
            return "Error: No browser or sampler provided to retrieve tool."
            
        urls = kwargs.get("url")
        query = kwargs.get("query")
        if not urls:
            return "Error: No url provided to retrieve tool."
        if not query:
            return "Error: No query provided to retrieve tool."

        # Support single URL as string
        if isinstance(urls, str):
            urls = [urls]

        all_page_contents = []
        errors = []

        # 1. Directly retrieve all page contents first
        for url in urls:
            # Filters out specific kwargs not intended for browser.do
            browser_params = {k: v for k, v in kwargs.items() if k not in ["url", "browser", "sampler", "query"]}
            
            response = browser.do("visit", url=url, **browser_params)
            
            if response.status.status == "success" and isinstance(response.data, PageContent):
                all_page_contents.append((url, browser.format_response_as_text(response)))
            else:
                errors.append(f"URL: {url} Error: {response.status.message}")
        
        if not all_page_contents:
            return "Error: Failed to retrieve content from any of the provided URLs.\n" + "\n".join(errors)

        # 2. Combine all retrieved content for a single extraction step
        combined_web_content = ""
        for url, content in all_page_contents:
            combined_web_content += f"\n--- Content from {url} ---\n{content}\n"
        
        # 3. Extract information using LLM in one go
        summary = await self._extract_info(sampler, combined_web_content, query)
        
        # Prepare the final observation
        observation = f"Summary of information extracted from {len(all_page_contents)} pages:\n{summary}"
        if errors:
            observation += "\n\nNote: Errors occurred for some URLs:\n" + "\n".join(errors)
            
        return observation
