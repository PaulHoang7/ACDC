import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def _resolve_vars(cfg: Dict[str, Any],
                  root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve ${key} references in config values."""
    if root is None:
        root = cfg
    resolved = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            resolved[k] = _resolve_vars(v, root)
        elif isinstance(v, str):
            def _replacer(m, _root=root):
                ref = m.group(1)
                parts = ref.split(".")
                val = _root
                for p in parts:
                    if isinstance(val, dict):
                        val = val[p]
                    else:
                        return m.group(0)
                return str(val)
            resolved[k] = re.sub(r"\$\{([^}]+)\}", _replacer, v)
        elif isinstance(v, list):
            resolved[k] = v
        else:
            resolved[k] = v
    return resolved


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file and resolve variable references."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    return _resolve_vars(cfg)


def load_paths() -> Dict[str, Any]:
    """Load the central paths config."""
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    return load_config(config_dir / "paths.yaml")


def load_all_configs() -> Dict[str, Any]:
    """Load and merge all config files."""
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    merged = {}
    for name in ["paths", "data", "model", "train"]:
        cfg_path = config_dir / f"{name}.yaml"
        if cfg_path.exists():
            merged[name] = load_config(cfg_path)
    return merged


def ensure_dirs(paths_cfg: Dict[str, Any]) -> None:
    """Create all output directories defined in paths config."""
    for key, val in paths_cfg.items():
        if isinstance(val, str) and ("dir" in key or key in ("processed",)):
            os.makedirs(val, exist_ok=True)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent
