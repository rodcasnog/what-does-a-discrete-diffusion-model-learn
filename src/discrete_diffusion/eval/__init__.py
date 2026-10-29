"""Evaluation entry points, each writing ``result.json`` to its Hydra output directory."""
from __future__ import annotations

import json
import os

from hydra.core.hydra_config import HydraConfig


def write_result(result: dict) -> None:
    """Print ``result`` and write it to ``result.json``."""
    print(json.dumps(result, indent=1))
    with open(os.path.join(HydraConfig.get().runtime.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
