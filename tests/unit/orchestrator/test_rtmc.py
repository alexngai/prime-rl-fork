"""RTMC: rollout-tree MC step credit (algo/rtmc.py + algo/rtmc_sig.py)."""

import asyncio
from unittest.mock import MagicMock

import pydantic
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage

from prime_rl.configs.algorithm import AlgoConfig
from prime_rl.orchestrator.algo import RTMCAlgorithm
from prime_rl.orchestrator.algo.rtmc import trace_steps
from prime_rl.orchestrator.algo.rtmc_sig import advantages, build_tree, coarsen_steps

_ALGO = pydantic.TypeAdapter(AlgoConfig)


def _node(message, *, parent, sampled, token_ids) -> MessageNode:
    return MessageNode(
        parent=parent,
        message=message,
        sampled=sampled,
        token_ids=token_ids,
        mask=[sampled] * len(token_ids),
        logprobs=[0.0] * len(token_ids) if sampled else [],
    )


def _episode(actions: list[str], reward: float, obs: str = "[command exited with status 0]") -> vf.Episode:
    """A linear trace: user prompt, then per action an assistant turn + tool observation."""
    nodes = [_node(UserMessage(content="U"), parent=None, sampled=False, token_ids=[1, 2])]
    tok = 3
    for action in actions:
        nodes.append(_node(
            AssistantMessage(content=f"THOUGHT\n```bash\n{action}\n```"),
            parent=len(nodes) - 1, sampled=True, token_ids=[tok, tok + 1]))
        nodes.append(_node(
            ToolMessage(tool_call_id="t", content=obs),
            parent=len(nodes) - 1, sampled=False, token_ids=[tok + 2]))
        tok += 3
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={"r": vf.Reward(score=reward)},
        ok=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(id="test-env", name="test-env"),
        task=trace.task,
        group=vf.GroupInfo(id="group"),
        traces=[trace],
    )


def test_config_registers_and_defaults():
    algo = _ALGO.validate_python({"type": "rtmc"})
    assert algo.type == "rtmc"
    assert algo.action_loss_type == "rl"
    assert algo.gamma == 0.99 and algo.n_prior == 2.0
    assert algo.content_hashes is False


def test_trace_steps_reads_graph_turns():
    ep = _episode(["cat core.py", "python -m pytest tests/t.py"], reward=1.0)
    steps = trace_steps(ep.traces[0])
    assert len(steps) == 2                        # aligned to trainable nodes
    assert steps[0] == ("||", "view:full@core.py")
    assert steps[1][1] == "test@tests/t.py:ok"    # observation read from the graph
    assert "core.py:Vf" in steps[1][0]


def test_score_group_assigns_signed_step_credit():
    # Two rollouts, same first action, divergent second action, opposite rewards.
    # NB: the second actions must differ in CATEGORY — under the default no-hash
    # coarsening, two modifies of the same file merge into one (s,a) node.
    ep_win = _episode(["cat core.py", "python -m pytest tests/t.py"], reward=1.0)
    ep_loss = _episode(["cat core.py", "rm core.py"], reward=0.0)
    algo = RTMCAlgorithm(_ALGO.validate_python({"type": "rtmc", "normalize": False}), MagicMock())
    asyncio.run(algo.score_group([ep_win, ep_loss]))
    win, loss = ep_win.traces[0], ep_loss.traces[0]
    win_advs = [a for node in win.nodes if node.advantages for a in node.advantages]
    loss_advs = [a for node in loss.nodes if node.advantages for a in node.advantages]
    assert len(win_advs) == 4 and len(loss_advs) == 4   # 2 turns x 2 tokens, broadcast
    assert win_advs[0] == win_advs[1]                    # uniform within a turn
    # Shared first (s,a): both rollouts' first turns share Q and V -> equal advantage;
    # gamma-discounting of the winner's return keeps it slightly below the smoothed V.
    assert abs(win_advs[0] - loss_advs[0]) < 1e-9
    # Divergent second turn: winner's action strictly out-credits the loser's.
    assert win_advs[2] > 0 > loss_advs[2]


def test_unanimous_group_zero_credit():
    eps = [_episode(["cat a.py"], reward=1.0), _episode(["cat b.py"], reward=1.0)]
    algo = RTMCAlgorithm(_ALGO.validate_python({"type": "rtmc"}), MagicMock())
    asyncio.run(algo.score_group(eps))
    for ep in eps:
        advs = [a for node in ep.traces[0].nodes if node.advantages for a in node.advantages]
        assert all(abs(a) < 1e-9 for a in advs)


def test_noop_turn_stays_aligned():
    ep = _episode(["cat core.py"], reward=1.0)
    # a turn with no bash block must still produce a step (noop), keeping 1:1 alignment
    nodes = ep.traces[0].nodes
    nodes.append(_node(AssistantMessage(content="just thinking"), parent=len(nodes) - 1,
                       sampled=True, token_ids=[90, 91]))
    steps = trace_steps(ep.traces[0])
    assert len(steps) == 2
    assert steps[1][1] == "noop"


def test_content_hash_toggle_uses_coarse_signatures():
    steps = [("a.py:M:a3f2+Vf||exec", "modify:b1c9@a.py")]
    assert coarsen_steps(steps) == [("a.py:M+Vf||exec", "modify@a.py")]
    # first-visit MC arithmetic (gamma=1): shared root, divergent outcomes
    group = [([("", "A")], 1.0), ([("", "B")], 0.0)]
    tree = build_tree(group, gamma=1.0)
    assert tree[("", "A")] == [1, 1.0] and tree[("", "B")] == [1, 0.0]
    adv = advantages(group, gamma=1.0, normalize=False)
    assert adv[0][0] == 0.5 and adv[1][0] == -0.5
