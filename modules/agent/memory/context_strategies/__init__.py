from .base import BaseContextStrategy
from .sliding_window import SlidingWindowStrategy
from .token_limit import TokenLimitStrategy

def get_context_strategy(
    strategy_name: str,
    **kwargs
) -> BaseContextStrategy:
    """
    Factory function to get the requested context strategy.
    
    Args:
        strategy_name: "sliding_window" or "token_limit"
        **kwargs: Arguments to pass to the strategy constructor
            (e.g., max_conversational_turns, max_tool_context, tokenizer_path, max_tokens)
    """
    if strategy_name == "sliding_window":
        return SlidingWindowStrategy(
            max_conversational_turns=kwargs.get("max_conversational_turns"),
            logger=kwargs.get("logger")
        )
    elif strategy_name == "token_limit":
        tokenizer_path = kwargs.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError("tokenizer_path is required for token_limit strategy")
            
        return TokenLimitStrategy(
            tokenizer_path=tokenizer_path,
            max_tokens=kwargs.get("max_tokens", 110 * 1024),
            max_tool_context=kwargs.get("max_tool_context"),
            reset_interval=kwargs.get("reset_interval"),
            logger=kwargs.get("logger")
        )
    else:
        raise ValueError(f"Unknown context strategy: {strategy_name}")
