from typing import List, Dict, Any


def format_predicates(predicates: List[Dict[str, Any]], max_content_length: int = None) -> str:
    """
    Format predicates for display.
    Shows for each predicate, numbered source references, and content (optionally truncated).

    Args:
        predicates: List of predicate dictionaries
        max_content_length: Maximum length of content to display (None for no truncation)

    Returns:
        Tuple of (formatted_string, link_mapping_dict)
        formatted_string: Formatted predicates with [*] and [0], [1] source references
        link_mapping_dict: Mapping from reference numbers to actual URLs
    """
    if not predicates:
        return "", {}

    formatted = []
    link_mapping = {}
    global_link_index = 0

    for pred in predicates:
        content = pred.get("content", "")
        topic = pred.get("topic", "General")
        links = pred.get("links", [])

        # Create numbered references for sources
        source_refs = []
        for link in links:
            if link not in link_mapping:
                link_mapping[link] = global_link_index
                global_link_index += 1
            source_refs.append(f"[{link_mapping[link]}]")

        links_str = f"Sources: {', '.join(source_refs)}" if source_refs else ""

        # Truncate content if max_content_length is specified
        if max_content_length is not None and len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        formatted.append(f"- {links_str}\n    Content: {content}")

    result = "\n\n".join(formatted)

    return result, link_mapping


def format_tools_description(tools: List[Dict[str, Any]]) -> str:
    """
    Format tool descriptions for system prompts.

    Args:
        tools: List of tool definitions in OpenAI function format

    Returns:
        Formatted string describing available tools
    """
    descriptions = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})

        param_descs = []
        for param_name, param_info in params.items():
            required = param_name in func.get("parameters", {}).get("required", [])
            param_descs.append(f"  - {param_name}: {param_info.get('description', '')}{' (required)' if required else ''}")

        descriptions.append(f"- {name}: {desc}\n" + "\n".join(param_descs))

    return "\n\n".join(descriptions)
