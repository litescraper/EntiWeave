import json as _json
from typing import Any, Optional

class JSONParser:
    """
    A parser to extract and parse JSON content from text.
    """
    def __init__(self):
        pass

    def parse(self, text: str) -> Optional[Any]:
        """
        Extract and parse the outermost JSON object/array in the given text.
        Returns the parsed Python object or None if not found/parsable.
        """
        if not text:
            return None
            
        # Quick path: if the whole text is valid JSON
        try:
            return _json.loads(text)
        except Exception:
            pass
            
        # Look for fenced ```json blocks first
        fence = "```json"
        idx = text.find(fence)
        if idx != -1:
            nl = text.find('\n', idx + len(fence))
            end = text.find("```", nl + 1 if nl != -1 else idx + len(fence))
            if nl != -1 and end != -1 and end > nl:
                inner = text[nl + 1:end]
                try:
                    return _json.loads(inner)
                except Exception:
                    # fall through to generic search
                    pass
                    
        # Generic outermost search
        span = self._find_outer_json_span(text)
        if span is None:
            return None
        start, end = span
        snippet = text[start:end]
        try:
            return _json.loads(snippet)
        except Exception:
            return None

    def _find_outer_json_span(self, text: str) -> Optional[tuple[int, int]]:
        """
        Find the span (start, end) of the outermost JSON object or array in text.
        """
        if not text:
            return None
        n = len(text)
        # Candidates are '{' or '['
        starts: list[int] = []
        for i, ch in enumerate(text):
            if ch == '{' or ch == '[':
                starts.append(i)
        
        # Try each start, prefer the earliest that yields a balanced parse
        for start in starts:
            stack: list[str] = []
            in_string = False
            escape = False
            for j in range(start, n):
                c = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        in_string = False
                    continue
                else:
                    if c == '"':
                        in_string = True
                        continue
                    if c == '{' or c == '[':
                        stack.append(c)
                    elif c == '}' or c == ']':
                        if not stack:
                            break
                        open_c = stack.pop()
                        if (open_c == '{' and c != '}') or (open_c == '[' and c != ']'):
                            break
                        if not stack:
                            # Balanced outermost JSON spans start..j
                            return (start, j + 1)
        return None

def extract_outer_json(text: str) -> Optional[Any]:
    """
    Convenience function for backward compatibility.
    """
    parser = JSONParser()
    return parser.parse(text)
