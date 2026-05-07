import sys
import io
import subprocess
from typing import Any, Dict, Union

# Child runs user code from stdin with the same preloaded modules as the in-process interpreter.
_CHILD_RUNNER = r"""import sys, io, traceback, math, json, datetime, re, os, base64, hashlib, builtins
code = sys.stdin.read()
out = io.StringIO()

def _print(*a, **k):
    k.setdefault("file", out)
    builtins.print(*a, **k)

g = {
    "math": math,
    "json": json,
    "datetime": datetime,
    "re": re,
    "os": os,
    "base64": base64,
    "hashlib": hashlib,
    "print": _print,
}
try:
    exec(compile(code, "<python_tool>", "exec"), g, g)
except Exception:
    sys.stderr.write(traceback.format_exc())
    sys.exit(1)
text = out.getvalue()
sys.stdout.write(text)
sys.exit(0)
"""


class PythonInterpreter:
    """
    A simple Python interpreter tool for executing code in a controlled environment.
    Code runs in a subprocess so wall-clock timeouts are enforced reliably (any thread;
    user code cannot swallow the timeout).
    """

    def __init__(self):
        self.globals: Dict[str, Any] = {}
        self._setup_initial_globals()
        self.output_buffer = io.StringIO()

    def _setup_initial_globals(self):
        import math
        import json
        import datetime
        import re
        import os
        import base64
        import hashlib

        self.globals.update({
            "math": math,
            "json": json,
            "datetime": datetime,
            "re": re,
            "os": os,
            "base64": base64,
            "hashlib": hashlib,
            "print": self._custom_print,
        })

    def _custom_print(self, *args, **kwargs):
        if 'file' not in kwargs:
            kwargs['file'] = self.output_buffer
        print(*args, **kwargs)

    def run(self, code: Any, timeout: Union[int, float] = 10) -> str:
        """
        Executes the provided Python code and returns the output from stdout or any error.
        Default wall-clock limit is 10 seconds; the child process is terminated on expiry.
        """
        if code is None:
            return "Error: No code provided to execute."

        if not isinstance(code, str):
            if isinstance(code, (list, tuple)):
                code = "\n".join(str(item) for item in code)
            else:
                code = str(code)

        self.output_buffer = io.StringIO()

        try:
            to = float(timeout)
        except (TypeError, ValueError):
            to = 10.0
        if to <= 0:
            to = 10.0

        try:
            completed = subprocess.run(
                [sys.executable, "-c", _CHILD_RUNNER],
                input=code,
                capture_output=True,
                text=True,
                timeout=to,
            )
        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {to} seconds."

        out = (completed.stdout or "").strip()
        err = (completed.stderr or "").strip()

        if completed.returncode != 0:
            if err:
                return f"Error during execution:\n{err}"
            return "Error during execution (process exited with errors; no traceback captured)."

        if not out:
            return "Code executed successfully (no output)."
        return completed.stdout

    def format_response_as_text(self, output: str) -> str:
        """
        Formats the output for the agent's observation.
        """
        return f"--- Python Output ---\n{output}\n---------------------"
