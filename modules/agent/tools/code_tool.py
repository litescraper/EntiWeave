from typing import Any, Dict, Optional
from .base_tool import BaseTool
from ..python.python_interpreter import PythonInterpreter

class CodeTool(BaseTool):
    """
    A tool that wraps the PythonInterpreter for executing code.
    """
    @property
    def name(self) -> str:
        return "python"

    @property
    def description(self) -> str:
        return "Execute Python code in a controlled environment."

    async def call(self, **kwargs: Any) -> str:
        """
        Executes the provided Python code using the passed interpreter.
        """
        interpreter = kwargs.get("interpreter")
        if not interpreter:
            return "Error: No interpreter provided to code tool."

        code = kwargs.get("code", "")
        if not code:
            # Fallback if code is passed differently
            if len(kwargs) == 1:
                # kwargs might contain 'interpreter' and one other thing
                other_keys = [k for k in kwargs.keys() if k != "interpreter"]
                if other_keys:
                    code = kwargs[other_keys[0]]
            elif "tool_name" in kwargs:
                other_keys = [k for k in kwargs.keys() if k not in ["tool_name", "interpreter"]]
                if other_keys:
                    code = kwargs[other_keys[0]]

        if not isinstance(code, str):
            code = str(code)

        timeout = kwargs.get("timeout", 10)
        output = interpreter.run(code, timeout=timeout)
        return interpreter.format_response_as_text(output)
