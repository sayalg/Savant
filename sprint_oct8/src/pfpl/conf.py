"""Config loading: a small Hydra-shaped layer over plain YAML.

PLAN.md Section 10 asks for Hydra. This implements the two Hydra features the plan
actually relies on - composable `defaults:` groups and dotted command-line
overrides - without the dependency, because a config framework that fails to import
on a compute node costs more than it saves on a two-week deadline. If Hydra is
wanted later, the YAML files already have its layout.

    cfg = load("config", ["model=variant_a", "plm.mode=esm2_bottleneck", "seed=3"])

Every run records `cfg.hash`, a content hash of the fully resolved config. That
hash is the key for the compute ledger (Section 7.1) and for cache reuse, and it is
what makes "which of the 300 configurations produced this number" answerable.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import CONFIGS


class Cfg(dict):
    """A dict with attribute access, because `cfg.model.plm.mode` reads better."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(f"no config key {k!r}; have {sorted(self)}") from None
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _yaml_load(p: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is not installed, and every config in this project is YAML.\n"
            "    pip install pyyaml") from None
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _coerce(s: str) -> Any:
    """'3' -> 3, 'true' -> True, 'null' -> None, '[1,2]' -> [1,2], else the string.

    Only YAML 1.2's null spellings map to None. NOT `none`: `plm.mode=none` is a
    real, named PLM mode (Section 6.7) and one of the most important cells in the
    ablation matrix, so coercing it to None would silently turn the single cheapest
    experiment into a crash. `yes`/`no` are left as strings for the same reason -
    surprising coercions in a CLI override are worse than a little verbosity.
    """
    low = s.strip().lower()
    if low in ("null", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _resolve_defaults(node: dict, base_dir: Path, seen: set[str],
                      self_dir: Path | None = None) -> dict:
    """Compose a `defaults:` list the way Hydra does: groups first, then this file.

    `_inherit: <name>` additionally merges a SIBLING file at the same level. Hydra's
    `defaults` puts a group file's contents under the group name, which is wrong for
    within-group inheritance: `configs/model/variant_b.yaml` inheriting variant_a
    would otherwise produce `model.model.*`. `_inherit` is the flat version.
    """
    merged: dict = {}
    for inh in _as_list(node.pop("_inherit", None)):
        path = (self_dir or base_dir) / f"{inh}.yaml"
        if not path.exists():
            raise SystemExit(f"_inherit: {inh!r} -> {path} not found")
        if str(path) in seen:
            raise SystemExit(f"circular _inherit at {path}")
        seen.add(str(path))
        merged = _deep_merge(merged, _resolve_defaults(_yaml_load(path), base_dir,
                                                       seen, path.parent))
    defaults = node.pop("defaults", []) or []
    for item in defaults:
        if isinstance(item, str):
            group, name = (item.split("=", 1) if "=" in item
                           else (None, item))  # bare filename
            rel = f"{group}/{name}" if group else name
        elif isinstance(item, dict):
            (group, name), = item.items()
            rel = f"{group}/{name}"
        else:
            raise SystemExit(f"cannot read defaults entry {item!r}")
        path = (base_dir / f"{rel}.yaml")
        if not path.exists():
            raise SystemExit(f"config group file not found: {path}")
        key = str(path)
        if key in seen:
            raise SystemExit(f"circular config defaults at {path}")
        seen.add(key)
        sub = _resolve_defaults(_yaml_load(path), base_dir, seen, path.parent)
        ## A group file's contents land under its group name (configs/model/x.yaml
        ## -> cfg.model.*), matching Hydra's package semantics for grouped configs.
        group_name = rel.split("/")[0] if "/" in rel else None
        merged = _deep_merge(merged, {group_name: sub} if group_name else sub)
    return _deep_merge(merged, node)


def config_hash(cfg: dict) -> str:
    """Stable over key order; ignores keys that do not change the computation."""
    d = copy.deepcopy(dict(cfg))
    for volatile in ("hash", "run_name", "overrides", "_meta"):
        d.pop(volatile, None)
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def load(name: str = "config", overrides: list[str] | None = None,
         base_dir: Path | None = None) -> Cfg:
    base_dir = Path(base_dir or CONFIGS)
    root = base_dir / f"{name}.yaml"
    if not root.exists():
        raise SystemExit(f"no config {root}")
    cfg = _resolve_defaults(_yaml_load(root), base_dir, {str(root)}, root.parent)

    applied: list[str] = []
    for ov in overrides or []:
        if "=" not in ov:
            raise SystemExit(f"override {ov!r} is not key=value")
        k, v = ov.split("=", 1)
        k = k.strip().lstrip("+")
        ## `model=variant_a` selects a GROUP FILE; `model.foo=1` sets a leaf. The
        ## difference is whether configs/<k>/<v>.yaml exists.
        grp = base_dir / k / f"{v}.yaml"
        if "." not in k and grp.exists():
            sub = _resolve_defaults(_yaml_load(grp), base_dir, {str(grp)}, grp.parent)
            ## Replace the group wholesale rather than merging onto the default one:
            ## `-o model=variant_c` must not leave variant_a's energy terms behind.
            cfg = _deep_merge(cfg, {k: sub}) if k not in cfg else {**cfg, k: sub}
        else:
            _set_dotted(cfg, k, _coerce(v))
        applied.append(ov)

    cfg["overrides"] = applied
    cfg["hash"] = config_hash(cfg)
    return Cfg(cfg)


def dump(cfg: Cfg, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(cfg), indent=1, default=str))


def add_cfg_args(ap) -> None:
    """Standard `--config` / `-o key=value` pair for every entry-point script."""
    ap.add_argument("--config", default="config", help="configs/<name>.yaml")
    ap.add_argument("-o", "--override", action="append", default=[], metavar="K=V",
                    help="dotted override, e.g. -o plm.mode=none -o seed=3")


def from_args(a) -> Cfg:
    return load(a.config, a.override)
