import copy
from typing import Dict, List

from rllm.agents.ttl_cum_agent import TTLCumulativeAgent


class TTLSlidingWindowAgent(TTLCumulativeAgent):
    """TTLCumulativeAgent with a sliding window on chat_completions.

    self.messages still grows unbounded (needed for trajectory/RL storage).
    Only the messages sent to the LLM are windowed.
    """

    def __init__(self, system_prompt, window_size=1, **kwargs):
        self.window_size = window_size
        super().__init__(system_prompt, **kwargs)

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        # self.messages = [system, u0, a0, u1, a1, ..., u_N]
        # Keep: system + last `window_size` (user, assistant) pairs + current pending user
        if self.window_size is None or self.window_size < 0:
            return copy.deepcopy(self.messages)  # no windowing

        system = self.messages[0]  # always keep
        rest = self.messages[1:]   # [u0, a0, u1, a1, ..., u_N]

        # Number of complete (user, assistant) pairs; last element is pending user
        n_pairs = (len(rest) - 1) // 2

        if n_pairs <= self.window_size:
            return copy.deepcopy(self.messages)

        # Keep last window_size pairs + pending user message
        keep_from = (n_pairs - self.window_size) * 2  # index into rest
        windowed = [system] + rest[keep_from:]
        return copy.deepcopy(windowed)
