"""RTMC signatures and rollout-tree Monte-Carlo advantages (arXiv:2604.11037).

Pure-stdlib logic shared by :class:`~prime_rl.orchestrator.algo.rtmc.RTMCAlgorithm`:
reduce each agent turn to an action signature and a cumulative, order-invariant state
signature; aggregate first-visit discounted returns across a group's rollouts in a tree
keyed by ``(state, action)``; read off ``A = Q(s,a) − V'(s)`` with the state value smoothed
toward the group success rate where visits are scarce (the paper's Eq. 11).

The paper's signature tables assume SWE-agent structured tools; this port classifies **bash
actions** (chorus's mini-swe-style harness). Provenance: `chorus/infra/rtmc_tree.py`, whose
feasibility probe on a 2,460-episode SWE-smith harvest (docs/55 gate G-P1) measured that
cross-rollout overlap is front-loaded (turns 0–15) and that dropping content hashes from
signatures (`coarsen_steps`) roughly doubles the informative band at turns 5–14 — hence the
algorithm's ``content_hashes = False`` default.

Deviations from the paper's tables, both on the side of MORE overlap:
- FLAGS holds coarse categories (``test:ok`` / ``test:err`` / ``submit`` / ``git:sub``),
  not content-hashed non-file actions.
- ``head``/``tail``/``less`` render as ``view:full`` rather than their own bucket notation.
- A turn with no parsed bash block is the explicit action ``noop`` (keeps the step list
  aligned 1:1 with the trace's trainable nodes).
"""

from __future__ import annotations

import hashlib
import math
import re

GAMMA = 0.99      # per-turn discount, the paper's setting
N_PRIOR = 2.0     # Eq. 11 smoothing pseudo-count

# --------------------------------------------------------------------------- signatures

_BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)(?:```|\Z)", re.S)
_SED_VIEW = re.compile(r"""sed\s+(?:-[a-zA-Z]*\s+)*-n\s+['"]?(\d+),(\d+)p['"]?\s+(\S+)""")
_REDIRECT = re.compile(r"(?<![>\d])>{1,2}\s*(\S+)")
_PATHISH = re.compile(r"^[\w./-]*(?:/[\w.-]+|\.\w{1,5})$")
_WRAPPERS = {"cd", "export", "set", "source", "sudo", "env", "time", "PYTHONPATH"}
_VIEWERS = {"cat", "head", "tail", "nl", "less", "more", "bat"}
_SEARCHERS = {"grep", "rg", "egrep", "fgrep", "find", "ls", "wc", "which", "tree"}
_TESTERS = {"pytest", "tox", "nose2", "unittest"}


def _hash4(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:4]


def _last_path(tokens: list[str]) -> str | None:
    for t in reversed(tokens):
        if not t.startswith("-") and _PATHISH.match(t):
            return t.rstrip(";")
    return None


def extract_bash(response: str) -> str | None:
    """The turn's action: all ```bash fenced code, joined (usually exactly one block)."""
    blocks = [b.strip() for b in _BASH_BLOCK.findall(response or "") if b.strip()]
    return "\n".join(blocks) if blocks else None


def action_signature(block: str | None) -> str:
    """``category[:detail]@target`` per RTMC Table 1, adapted to a bash harness."""
    if block is None or not block.strip():
        return "noop"
    text = block.strip()
    if "FINAL_OUTPUT" in text or "SUBMIT" in text.upper():
        return "submit"
    m = _SED_VIEW.search(text)
    if m and "-i" not in text.split():
        a, b = int(m.group(1)) // 100, int(m.group(2)) // 100
        return f"view:part[{a}-{b}]@{m.group(3)}"

    words = []
    for chunk in re.split(r"[;&|]+", text.replace("\n", ";")):
        toks = chunk.split()
        # Strip leading env assignments (PYTHONPATH=... cmd) and wrappers WITHIN the
        # chunk — skipping the whole chunk dumped every env-prefixed command into
        # `other` (measured on s2gate3: 35 PYTHONPATH-prefixed test/exec commands).
        # `cd` consumes its directory argument; a chunk it exhausts falls through to
        # the next one (so `cd /testbed && pytest` still classifies as the pytest).
        while toks:
            if "=" in toks[0] and "/" not in toks[0].split("=", 1)[0]:
                toks = toks[1:]
            elif toks[0] == "cd":
                toks = toks[2:] if len(toks) > 1 else []
            elif toks[0] in _WRAPPERS:
                toks = toks[1:]
            else:
                break
        if toks and not toks[0].startswith("-"):
            words = toks
            break
    if not words:
        # A lone wrapper (bare `cd /testbed`) navigates without acting.
        return "noop"
    # Basename the command so absolute interpreters classify (/usr/bin/python3,
    # /usr/local/bin/pytest — 70+ occurrences in s2gate3 all landed in `other`).
    cmd = words[0].rsplit("/", 1)[-1]
    if cmd in ("pip", "pip3", "uv", "conda", "apt-get", "apt"):
        return "pkg"

    redir = _REDIRECT.search(text)
    if redir and cmd != "grep":
        return f"modify:{_hash4(text)}@{redir.group(1)}"
    if cmd in _TESTERS or (cmd.startswith("python") and ("-m pytest" in text or "-m unittest" in text)):
        return f"test@{_last_path(words[1:]) or 'all'}"
    if cmd.startswith("python"):
        writes = ("open(" in text and re.search(r"""open\([^)]*['"][wa]""", text)) or ".write" in text
        if "<<" in text or "-c" in words:
            return (f"modify:{_hash4(text)}@inline" if writes else f"exec:{_hash4(text)}")
        tgt = _last_path(words[1:])
        return f"test@{tgt}" if tgt and "test" in tgt else f"exec:{_hash4(text)}"
    if cmd in _VIEWERS:
        return f"view:full@{_last_path(words[1:]) or '?'}"
    if cmd in _SEARCHERS:
        tgt = _last_path(words[1:])
        return f"search@{tgt}" if tgt else "search"
    if cmd == "sed" and "-i" in words:
        return f"modify:{_hash4(text)}@{_last_path(words[1:]) or '?'}"
    if cmd in ("touch", "mkdir"):
        return f"create@{_last_path(words[1:]) or '?'}"
    if cmd in ("cp", "mv", "rm", "patch", "tee"):
        return f"modify:{_hash4(text)}@{_last_path(words[1:]) or '?'}"
    if cmd == "git":
        sub = words[1] if len(words) > 1 else "?"
        return f"modify:{_hash4(text)}@git" if sub in ("apply", "checkout", "revert") else f"git:{sub}"
    return f"other:{_hash4(text)}"


_RC = re.compile(r"(?:command exited with status|returncode|exit code|Exit)\D{0,3}(-?\d+)", re.I)


def observation_result(obs: str) -> str:
    m = _RC.search(obs or "")
    return "?" if not m else ("ok" if m.group(1) == "0" else "err")


def apply_action(state: dict, asig: str) -> None:
    """Fold one action into the cumulative state: per-file op sets + a FLAGS set."""
    cat, _, rest = asig.partition("@")
    target = rest or None
    kind = cat.split(":")[0]
    op = {"view": "V" + (cat[9:] if cat.startswith("view:part") else "f"),
          "search": "S", "create": "C"}.get(kind)
    if kind == "modify":
        op = "M:" + cat.split(":")[1]
    if op and target and target not in ("?", "inline", "git", "all"):
        state.setdefault(target, set()).add(op)
    else:
        if kind == "test":  # test@x:ok -> "test:ok"; result-less -> "test"
            flag = "test:" + rest.rsplit(":", 1)[1] if rest and ":" in rest else "test"
        else:
            flag = {"submit": "submit", "git": cat, "exec": "exec",
                    "other": "other", "search": "S", "noop": "noop", "pkg": "pkg"}.get(kind, kind)
        state.setdefault("_flags", set()).add(flag)


def state_signature(state: dict) -> str:
    files = sorted(k for k in state if k != "_flags")
    body = "|".join(f"{f}:{'+'.join(sorted(state[f]))}" for f in files)
    flags = "+".join(sorted(state.get("_flags", ())))
    return f"{body}||{flags}"


_HASH_OP = re.compile(r"\b(M|I|modify|other|exec):[0-9a-f]{4}\b")


def coarsen_steps(steps: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop content hashes from signatures: ``M:a3f2`` → ``M``.

    Content hashes are a state-space divergence engine — two rollouts editing the same
    file with different bytes never share a state again. Coarsening to file level merges
    them ("has edited core.py"), trading state fidelity for cross-rollout overlap depth.
    Measured on the chorus harvest (docs/55 G-P1): roughly doubles the informative band
    at turns 5–14. The algorithm's default.
    """
    return [(_HASH_OP.sub(r"\1", s), _HASH_OP.sub(r"\1", a)) for s, a in steps]


# --------------------------------------------------------------------------- tree + A

def build_tree(group: list[tuple[list[tuple[str, str]], float]], gamma: float = GAMMA) -> dict:
    """First-visit MC statistics over one group: ``{(s, a): [count, return_sum]}``."""
    tree: dict = {}
    for steps, reward in group:
        seen: set = set()
        T = len(steps)
        for t, (s, a) in enumerate(steps):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            n_sum = tree.setdefault((s, a), [0, 0.0])
            n_sum[0] += 1
            n_sum[1] += gamma ** (T - 1 - t) * reward
    return tree


def advantages(group: list[tuple[list[tuple[str, str]], float]], gamma: float = GAMMA,
               n_prior: float = N_PRIOR, normalize: bool = True) -> list[list[float]]:
    """Per-rollout, per-step ``A(s,a) = Q(s,a) − V'(s)``, ``V'`` smoothed toward the
    group success rate (Eq. 11). ``normalize`` divides by the group std of A —
    sign-preserving, no centering, per the paper's optional per-group normalization."""
    tree = build_tree(group, gamma)
    by_state: dict = {}
    for (s, a), (n, tot) in tree.items():
        acc = by_state.setdefault(s, [0, 0.0])
        acc[0] += n
        acc[1] += tot
    v_prior = (sum(r for _, r in group) / len(group)) if group else 0.0
    out = []
    for steps, _ in group:
        row = []
        for s, a in steps:
            n_sa, sum_sa = tree[(s, a)]
            n_s, sum_s = by_state[s]
            v_hat = sum_s / n_s
            v_sm = (n_s * v_hat + n_prior * v_prior) / (n_s + n_prior)
            row.append(sum_sa / n_sa - v_sm)
        out.append(row)
    if normalize:
        flat = [x for row in out for x in row]
        mu = sum(flat) / len(flat) if flat else 0.0
        std = math.sqrt(sum((x - mu) ** 2 for x in flat) / len(flat)) if flat else 0.0
        if std > 1e-8:
            out = [[x / std for x in row] for row in out]
    return out
