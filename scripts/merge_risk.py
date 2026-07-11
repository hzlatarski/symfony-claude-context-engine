"""Merge-order conflict risk across in-flight branches, via graph communities.

Given a chronically-dirty ``main`` and several feature branches / worktrees
in flight at once (this project's normal state), the question that actually
bites at merge time is: *which branches, if merged in the wrong order, will
collide?* Two signals answer it:

1. **Direct file conflicts** — the same file changed on 2+ branches. The
   hard, unambiguous signal: git itself will conflict.
2. **Community overlap** — two branches touch *different* files that live in
   the same semantic community (Leiden cluster over the unified graph). Not
   a textual conflict, but a coupling risk: changes to tightly-related code
   are the ones most likely to interact semantically even without a merge
   conflict. This is the signal graphify's ``prs --conflicts`` surfaces and
   that plain ``git`` can't see.

The core (`compute`) is pure — it takes a ``{branch: [files]}`` map, the
unified graph, and the community partition, and returns structured risk. The
git plumbing lives in the thin MCP wrapper (`mcp_server._build_merge_order_risk`)
so this module is fully testable without a repo.
"""
from __future__ import annotations

from itertools import combinations


def _file_node(rel_path: str) -> str:
    return f"file:{rel_path.replace(chr(92), '/')}"


def build_node_community_map(clusters: list[dict]) -> dict[str, tuple[int, str]]:
    """``node_id -> (community_id, label)`` from Leiden community records."""
    out: dict[str, tuple[int, str]] = {}
    for c in clusters:
        cid = c.get("community_id", -1)
        label = c.get("label", "")
        for nid in c.get("members", []):
            out[nid] = (cid, label)
    return out


def compute(
    branch_files: dict[str, list[str]],
    node_community: dict[str, tuple[int, str]],
    ancestry: set[tuple[str, str]] | None = None,
) -> dict:
    """Compute direct + community-overlap merge risk between branches.

    Args:
        branch_files: ``{branch_name: [changed file rel-paths]}``.
        node_community: ``node_id -> (community_id, label)`` (from
            ``build_node_community_map``).
        ancestry: set of ``(ancestor, descendant)`` branch pairs. When branch
            B is stacked on A, ``diff base...B`` restates all of A's files, so
            naively they'd look like conflicts — but merging A then B is a
            clean fast-forward, not a conflict. Passing ancestry lets the
            function drop those false positives: a file's "conflict" branch
            set is reduced to its independent *tips*, and overlap pairs in an
            ancestor relation are skipped entirely.

    Returns a dict::

        {
          "branches": [{"branch", "file_count", "community_count"}],
          "direct_conflicts": [{"file", "branches": [...]}],
          "community_overlaps": [
             {"branch_a", "branch_b", "shared": [
                 {"community_id", "label", "files_a": [...], "files_b": [...]}
             ]}
          ],
        }
    """
    ancestry = ancestry or set()

    def _is_ancestor_pair(a: str, b: str) -> bool:
        """True if a and b are in an ancestor/descendant relation (either way)."""
        return (a, b) in ancestry or (b, a) in ancestry

    # Per-branch: file set + community → files map.
    branch_communities: dict[str, dict[int, dict]] = {}
    per_branch_summary: list[dict] = []
    file_to_branches: dict[str, set[str]] = {}

    for branch, files in branch_files.items():
        comm_files: dict[int, dict] = {}
        for rel in files:
            file_to_branches.setdefault(rel, set()).add(branch)
            hit = node_community.get(_file_node(rel))
            if hit is None:
                continue
            cid, label = hit
            bucket = comm_files.setdefault(cid, {"label": label, "files": []})
            bucket["files"].append(rel)
        branch_communities[branch] = comm_files
        per_branch_summary.append({
            "branch": branch,
            "file_count": len(files),
            "community_count": len(comm_files),
        })

    # Direct conflicts: a file changed by 2+ *independent* branches. Reduce
    # each file's branch set to its tips — drop any branch that is an ancestor
    # of another branch in the same set (a stacked chain restates the parent's
    # files but merges cleanly in order). A conflict remains only if 2+
    # independent tips still touch the file.
    def _tips(brs: set[str]) -> list[str]:
        return sorted(
            b for b in brs
            if not any(b != o and (b, o) in ancestry for o in brs)
        )

    direct = []
    for rel, brs in file_to_branches.items():
        if len(brs) < 2:
            continue
        tips = _tips(brs)
        if len(tips) >= 2:
            direct.append({"file": rel, "branches": tips})
    direct.sort(key=lambda d: (-len(d["branches"]), d["file"]))

    # Community overlaps between each independent branch pair (stacked pairs
    # skipped — their overlap is expected, not a risk).
    overlaps: list[dict] = []
    for a, b in combinations(sorted(branch_files), 2):
        if _is_ancestor_pair(a, b):
            continue
        ca, cb = branch_communities[a], branch_communities[b]
        shared_cids = sorted(set(ca) & set(cb))
        shared: list[dict] = []
        for cid in shared_cids:
            files_a = sorted(ca[cid]["files"])
            files_b = sorted(cb[cid]["files"])
            # A community shared only because both branches touch the exact
            # same file is already reported as a direct conflict — skip those
            # that add no *additional* coupling signal.
            if set(files_a) == set(files_b) and len(files_a) == 1:
                continue
            shared.append({
                "community_id": cid,
                "label": ca[cid]["label"] or cb[cid]["label"],
                "files_a": files_a,
                "files_b": files_b,
            })
        if shared:
            overlaps.append({"branch_a": a, "branch_b": b, "shared": shared})

    overlaps.sort(key=lambda o: (-len(o["shared"]), o["branch_a"], o["branch_b"]))

    return {
        "branches": per_branch_summary,
        "direct_conflicts": direct,
        "community_overlaps": overlaps,
    }


def render(result: dict, base: str) -> str:
    """Render ``compute()`` output as a markdown risk report."""
    branches = result["branches"]
    direct = result["direct_conflicts"]
    overlaps = result["community_overlaps"]

    if not branches:
        return (
            f"No branches found ahead of `{base}`. Nothing in flight to compare — "
            f"pass explicit branch names if you expected some."
        )

    lines = [
        "# Merge-order conflict risk",
        f"- Base: `{base}`",
        f"- Branches compared: {len(branches)}",
        f"- Direct file conflicts: {len(direct)}",
        f"- Community-overlap pairs: {len(overlaps)}",
        "",
        "## Branches in flight",
    ]
    for b in sorted(branches, key=lambda x: -x["file_count"]):
        lines.append(
            f"- `{b['branch']}` — {b['file_count']} file(s), "
            f"{b['community_count']} community(ies)"
        )

    lines.append("")
    lines.append(f"## ⛔ Direct file conflicts ({len(direct)})")
    if direct:
        lines.append("Same file changed on multiple branches — git *will* conflict. "
                     "Merge these branches back-to-back and rebase the rest.")
        for d in direct:
            lines.append(f"- `{d['file']}` — {', '.join(f'`{x}`' for x in d['branches'])}")
    else:
        lines.append("_None — no file is touched by more than one branch._")

    lines.append("")
    lines.append(f"## ⚠ Community-overlap risk ({len(overlaps)} pair(s))")
    if overlaps:
        lines.append("Different files, same semantic cluster — no textual conflict, "
                     "but a coupling risk. Review these pairs together and prefer "
                     "merging the smaller change first.")
        for o in overlaps:
            lines.append("")
            lines.append(f"### `{o['branch_a']}` ↔ `{o['branch_b']}` "
                         f"({len(o['shared'])} shared community(ies))")
            for s in o["shared"]:
                lines.append(f"- **{s['label']}** (community {s['community_id']})")
                lines.append(f"  - {o['branch_a']}: " + ", ".join(f"`{f}`" for f in s["files_a"][:6]))
                lines.append(f"  - {o['branch_b']}: " + ", ".join(f"`{f}`" for f in s["files_b"][:6]))
    else:
        lines.append("_None — no two branches touch the same semantic cluster._")

    return "\n".join(lines)
