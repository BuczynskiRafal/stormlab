"""System checks: the model files the application serves must be present."""

from __future__ import annotations

import os
from typing import Any

from django.core.checks import CheckMessage, Error, register


@register()
def model_files_present(app_configs: Any, **kwargs: Any) -> list[CheckMessage]:
    """Report every missing model file, so a broken checkout fails at startup instead of serving no recommendations."""
    from .core import predictor

    required = [predictor.catchment_classifier_path]
    if os.environ.get("GNN_ENABLED") is None or predictor.gnn_enabled():
        imputation_path = os.path.join(os.path.dirname(predictor.gnn_spec_path), predictor.GNN_CONFIG["imputation"])
        required += [*predictor.gnn_model_paths, predictor.gnn_spec_path, imputation_path]

    missing = [path for path in required if not os.path.isfile(path)]
    if not missing:
        return []
    return [
        Error(
            f"Model files are missing: {', '.join(missing)}",
            hint="Restore them from the repository; they are tracked in git.",
            id="sa.E001",
        )
    ]
