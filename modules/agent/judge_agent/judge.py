import os
import re
import asyncio
from typing import Dict, Any, Optional
from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, AgentRunResult
from modules.agent.memory.base_memory import BaseMemory

class JudgeAgent(BaseLLMAgent):
    """
    An agent that judges whether a predicted answer matches a reference answer.
    """

    def __init__(
        self,
        config: Optional[AgentSamplerConfig] = None,
        voting_n: int = 1,
    ):
        """
        Initialize the Judge Agent.
        """
        # Load prompts from yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompt.yaml")
        prompt_data = self._load_prompts(yaml_path)

        system_prompt = prompt_data.get("system_prompt", "You are an expert judge.")
        self.judge_template = prompt_data.get("judge_prompt", "[Question]: {question}\n[Reference Answer]: {reference_answer}\n[Predicted Response]: {predicted_answer}")

        super().__init__(
            config=config,
            system_prompt=system_prompt
        )
        self.voting_n = voting_n

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        The Judge Agent doesn't use tools.
        """
        raise NotImplementedError("JudgeAgent does not use tools.")

    async def run(self, question: str, predicted_answer: str, reference_answer: str) -> AgentRunResult:
        """
        Judge if the predicted answer matches the reference answer.

        Args:
            question: The original question.
            predicted_answer: The answer provided by another LLM.
            reference_answer: The correct ground truth answer.

        Returns:
            An AgentRunResult containing the judgment.
        """
        if self.voting_n <= 1:
            return await self._run_once(question, predicted_answer, reference_answer)

        # Run multiple times for voting in parallel
        tasks = [self._run_once(question, predicted_answer, reference_answer) for _ in range(self.voting_n)]
        results = await asyncio.gather(*tasks)

        # Count votes for 'correct' (True/False)
        true_results = [r for r in results if r.metadata.get("correct") is True]
        false_results = [r for r in results if r.metadata.get("correct") is False]

        # Determine winner
        if len(true_results) >= len(false_results):
            winner_results = true_results
            final_correct = True
        else:
            winner_results = false_results
            final_correct = False

        # Select the one reasoning from the winning (take the first one)
        representative_result = winner_results[0] if winner_results else results[0]
        
        final_result = AgentRunResult(
            answer=str(final_correct),
            trajectory=representative_result.trajectory,
            status="success",
            metadata=representative_result.metadata.copy()
        )
        final_result.metadata.update({
            "voting_n": self.voting_n,
            "votes_true": len(true_results),
            "votes_false": len(false_results),
            "all_judgments": [r.metadata.get("correct") for r in results]
        })
        # Ensure final_correct is accurately reflected
        final_result.metadata["correct"] = final_correct
        
        return final_result

    async def _run_once(self, question: str, predicted_answer: str, reference_answer: str) -> AgentRunResult:
        """
        Single run of the judgment process.
        """
        user_prompt = self.judge_template.format(
            question=question,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer
        )

        messages = [{"role": "user", "content": user_prompt}]
        
        max_attempts = 3
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                # Generate response from LLM
                formatted_messages = BaseMemory.format_messages(messages, sampler_type=self.config.sampler_type, include_thinking=True)
                response = await asyncio.to_thread(self.generate_response, formatted_messages)
                content = response.content

                # Parse the output
                extracted_answer = None
                reasoning = None
                correct = None

                # Use regex to extract fields from the LLM's response
                # We use DOTALL for extracted_answer and reasoning in case they span multiple lines,
                # but we need to be careful not to consume the next field.
                # We look ahead for either a newline followed by the next tag, or just the next tag.
                match_extracted = re.search(r"extracted_answer:\s*(.*?)(?=\s*\nreasoning:|\s*\ncorrect:|\s*reasoning:|\s*correct:|$)", content, re.IGNORECASE | re.DOTALL)
                if match_extracted:
                    extracted_answer = match_extracted.group(1).strip()
                    
                match_reasoning = re.search(r"reasoning:\s*(.*?)(?=\s*\ncorrect:|\s*\nextracted_answer:|\s*correct:|\s*extracted_answer:|$)", content, re.IGNORECASE | re.DOTALL)
                if match_reasoning:
                    reasoning = match_reasoning.group(1).strip()
                    
                match_correct = re.search(r"correct:\s*(true|false|yes|no)", content, re.IGNORECASE)
                if match_correct:
                    val = match_correct.group(1).lower()
                    correct = val in ["true", "yes"]

                # Strict validation: we need at least 'correct' and 'reasoning'
                if correct is None or not reasoning:
                    raise ValueError(f"Failed to extract 'correct' and 'reasoning' from response. Attempt {attempt + 1}/{max_attempts}")

                assistant_message = {"role": "assistant", "content": content}
                if response.reasoning_content:
                    assistant_message["reasoning_content"] = response.reasoning_content

                return AgentRunResult(
                    answer=str(correct),
                    trajectory=[
                        {"role": "user", "content": user_prompt},
                        assistant_message
                    ],
                    status="success",
                    metadata={
                        "correct": correct,
                        "reasoning": reasoning,
                        "extracted_answer": extracted_answer or "None",
                        "raw_response": content
                    }
                )
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    # Add a small hint to the messages to help the model correct itself? 
                    # No, usually just resmapling with the same prompt is enough if it's a transient failure,
                    # or we could append a message saying "Please follow the format...".
                    # Let's try just resampling first as requested ("request another sampling").
                    continue
                else:
                    break
        
        # If all attempts fail, return a failure result or raise the last exception
        # Returning a failure result is better for the pipeline
        return AgentRunResult(
            answer="False",
            trajectory=[{"role": "user", "content": user_prompt}],
            status="error",
            metadata={
                "correct": False,
                "reasoning": f"Failed to parse judgment after {max_attempts} attempts: {str(last_exception)}",
                "error": str(last_exception)
            }
        )
