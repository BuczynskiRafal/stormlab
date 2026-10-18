import hashlib
import json
import os
import sys
import logging
import threading
from functools import lru_cache
from typing import Optional, Any, Dict

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logger = logging.getLogger(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__))
catchment_classifier_path = os.path.join(current_directory, "catchment_classifier", "model.keras")
recommendations_classifier_path = os.path.join(current_directory, "recommendations", "recomendations.keras")
GNN_CONFIG = {
    "weights": os.environ.get("GNN_WEIGHTS", "gnn_c3c_winner_v1.weights.h5"),
    "spec": os.environ.get("GNN_SPEC", "gnn_c3c_winner_v1.spec.json"),
    "max_hops": int(os.environ.get("GNN_MAX_HOPS", "2")),
}
gnn_model_path = os.path.join(current_directory, "recommendations", GNN_CONFIG["weights"])
gnn_spec_path = os.path.join(current_directory, "recommendations", GNN_CONFIG["spec"])


def gnn_enabled() -> bool:
    raw = os.environ.get("GNN_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return os.path.isfile(gnn_model_path)


# Lazy-loaded models
_classifier: Optional[Any] = None
_recommendation: Optional[Any] = None
_gnn_recommendation: Optional[Any] = None
_models_loaded = False
_load_lock = threading.Lock()


@lru_cache(maxsize=8)
def _sha256_file_safe(path: str) -> Optional[str]:
    try:
        if not os.path.isfile(path):
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


@lru_cache(maxsize=8)
def _read_gnn_spec_contracts(spec_path: str) -> tuple[str, str]:
    feature_contract = "v4_physical"
    target_contract = "primary_action_v3"
    try:
        if os.path.isfile(spec_path):
            with open(spec_path, "r", encoding="utf-8") as fh:
                spec_data = json.load(fh)
            feature_contract = spec_data.get("feature_contract", feature_contract)
            target_contract = spec_data.get("target_contract", target_contract)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to read GNN spec metadata from {spec_path}: {e}")
    return feature_contract, target_contract


def get_model_metadata(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Return provenance metadata for the active recommendation model."""
    enabled = gnn_enabled()
    resolved_name = model_name if model_name is not None else ("GNN" if enabled else "MLP")
    is_gnn = str(resolved_name).upper() == "GNN" and enabled
    if is_gnn:
        weights_name = GNN_CONFIG["weights"]
        model_id = weights_name[:-11] if weights_name.endswith(".weights.h5") else os.path.splitext(weights_name)[0]
        feature_contract, target_contract = _read_gnn_spec_contracts(gnn_spec_path)
        return {
            "model_kind": "gnn",
            "model_id": model_id,
            "weights_sha256": _sha256_file_safe(gnn_model_path),
            "feature_contract": feature_contract,
            "target_contract": target_contract,
            "gnn_enabled": enabled,
        }
    return {
        "model_kind": "mlp",
        "model_id": "mlp_historical_20250615",
        "weights_sha256": _sha256_file_safe(recommendations_classifier_path),
        "feature_contract": "v1_historical_33",
        "target_contract": "historical_9class",
        "gnn_enabled": enabled,
    }


def _load_models() -> None:
    """Lazy load TensorFlow models on first access.

    _models_loaded means the load attempt finished; model availability is exposed by getters.
    """
    global _classifier, _recommendation, _gnn_recommendation, _models_loaded

    if _models_loaded:
        return

    with _load_lock:
        if _models_loaded:
            return

        try:
            # Disable GPU for consistency
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            from tensorflow.keras.models import load_model

            try:
                _classifier = load_model(catchment_classifier_path)
                logger.info(f"Catchment classifier loaded from: {catchment_classifier_path}")
            except (FileNotFoundError, ValueError, OSError):
                logger.error(f"Cannot load catchment classifier: {catchment_classifier_path}")

            try:
                _recommendation = load_model(recommendations_classifier_path)
                logger.info(f"MLP recommendation model loaded from: {recommendations_classifier_path}")
            except (FileNotFoundError, ValueError, OSError):
                logger.error(f"Cannot load MLP recommendation model: {recommendations_classifier_path}")

            if gnn_enabled():
                try:
                    from .graph_constructor import load_gnn_model_weights

                    _gnn_recommendation = load_gnn_model_weights(gnn_model_path)
                    if _gnn_recommendation:
                        logger.info(f"GNN model loaded successfully from {gnn_model_path}")
                    else:
                        logger.warning("GNN model loading failed")
                except Exception as e:
                    logger.warning(f"GNN model loading failed: {e}")
            else:
                logger.info("GNN model loading skipped because GNN_ENABLED is false.")

        except ImportError:
            logger.warning("TensorFlow not available - models will not be loaded")
        finally:
            _models_loaded = True


def get_classifier() -> Optional[Any]:
    """Get the catchment classifier model."""
    _load_models()
    return _classifier


def get_recommendation() -> Optional[Any]:
    """Get the MLP recommendation model."""
    _load_models()
    return _recommendation


def get_gnn_recommendation() -> Optional[Any]:
    """Get the GNN recommendation model."""
    _load_models()
    return _gnn_recommendation


# Backward compatibility - lazy properties
class _LazyModelProxy:
    def __init__(self, getter):
        self._getter = getter

    def __getattr__(self, name):
        model = self._getter()
        if model is None:
            raise RuntimeError("Model not loaded - TensorFlow may not be available")
        return getattr(model, name)

    def __call__(self, *args, **kwargs):
        model = self._getter()
        if model is None:
            raise RuntimeError("Model not loaded - TensorFlow may not be available")
        return model(*args, **kwargs)


classifier = _LazyModelProxy(get_classifier)
recommendation = _LazyModelProxy(get_recommendation)
gnn_recommendation = _LazyModelProxy(get_gnn_recommendation)

__all__ = [
    "classifier",
    "recommendation",
    "gnn_recommendation",
    "get_classifier",
    "get_recommendation",
    "get_gnn_recommendation",
    "get_model_metadata",
    "GNN_CONFIG",
    "gnn_enabled",
]
