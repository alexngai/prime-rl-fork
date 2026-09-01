from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import RTMCAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.algo.rtmc_sig import (
    action_signature,
    advantages,
    apply_action,
    coarsen_steps,
    extract_bash,
    observation_result,
    state_signature,
)

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def _text(message: vf.Message | None) -> str:
    """Message content as plain text (multimodal parts reduced to their text)."""
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if text:
            parts.append(str(text))
    return " ".join(parts)


def trace_steps(trace: vf.Trace) -> list[tuple[str, str]]:
    """One ``(state_signature_before, action_signature)`` per trainable node, in order.

    Turn structure is read straight off the trace graph: nodes with any sampled token are
    the agent's turns (the exact filter :func:`assign_advantages` uses, so the step list
    aligns 1:1 with the advantage stream); the non-sampled nodes that follow a turn carry
    its observation. A turn without a parsed bash block is the explicit ``noop`` action.
    """
    steps: list[tuple[str, str]] = []
    state: dict = {}
    nodes = trace.nodes
    turn_idx = [i for i, node in enumerate(nodes) if any(node.mask)]
    for j, i in enumerate(turn_idx):
        asig = action_signature(extract_bash(_text(nodes[i].message)))
        if asig.startswith(("test@", "exec")):
            nxt = turn_idx[j + 1] if j + 1 < len(turn_idx) else len(nodes)
            obs = " ".join(_text(n.message) for n in nodes[i + 1:nxt])
            if asig.startswith("test@"):
                asig = f"{asig}:{observation_result(obs)}"
        steps.append((state_signature(state), asig))
        apply_action(state, asig)
    return steps


class RTMCAlgorithm(Algorithm):
    """Rollout-tree Monte-Carlo step credit (RTMC, arXiv:2604.11037): aggregate the
    group's rollouts in a signature-keyed tree, credit each turn with
    ``Q(s,a) − V'(s)`` from first-visit MC returns, and broadcast the turn's advantage
    over its action tokens. Critic-free, fork-free, zero extra rollouts over GRPO —
    only the credit assignment changes."""

    def __init__(self, config: RTMCAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.gamma = config.gamma
        self.n_prior = config.n_prior
        self.normalize = config.normalize
        self.content_hashes = config.content_hashes

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if not traces:
            return
        group = []
        for trace in traces:
            steps = trace_steps(trace)
            if not self.content_hashes:
                steps = coarsen_steps(steps)
            group.append((steps, float(trace.reward)))
        advs = advantages(group, gamma=self.gamma, n_prior=self.n_prior, normalize=self.normalize)
        for trace, (steps, _), row in zip(traces, group, advs, strict=True):
            values: list[float] = []
            turns = [node for node in trace.nodes if any(node.mask)]
            for node, advantage in zip(turns, row, strict=True):
                values.extend([advantage] * sum(node.mask))
            assign_advantages(trace, values)
