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
# The GNN is served as the mean of the class probabilities of five training runs.
# GNN_WEIGHTS may name other members as a comma-separated list of file names.
_DEFAULT_GNN_WEIGHTS = ",".join(f"gnn_recommender.run{run}.weights.h5" for run in range(1, 6))
GNN_CONFIG = {
    "model_id": os.environ.get("GNN_MODEL_ID", "gnn_recommender_v1"),
    "weights": tuple(name.strip() for name in os.environ.get("GNN_WEIGHTS", _DEFAULT_GNN_WEIGHTS).split(",") if name.strip()),
    "spec": os.environ.get("GNN_SPEC", "gnn_recommender.spec.json"),
    "imputation": os.environ.get("GNN_IMPUTATION", "gnn_recommender.imputation.json"),
    "max_hops": int(os.environ.get("GNN_MAX_HOPS", "2")),
}
gnn_model_paths = tuple(os.path.join(current_directory, "recommendations", name) for name in GNN_CONFIG["weights"])
gnn_spec_path = os.path.join(current_directory, "recommendations", GNN_CONFIG["spec"])


def gnn_enabled() -> bool:
    """GNN_ENABLED if set; otherwise on when any configured member file exists.

    Any, not all: a partly missing ensemble must fail to load loudly rather than
    quietly switch serving off.
    """
    raw = os.environ.get("GNN_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return any(os.path.isfile(path) for path in gnn_model_paths)


def _ensemble_weights_sha256(paths: tuple[str, ...]) -> Optional[str]:
    """sha256 over the concatenated hex sha256 digests of the member files, in run order."""
    digests = [digest for digest in (_sha256_file_safe(path) for path in paths) if digest is not None]
    if not digests or len(digests) != len(paths):
        return None
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


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
        feature_contract, target_contract = _read_gnn_spec_contracts(gnn_spec_path)
        return {
            "model_kind": "gnn",
            "model_id": GNN_CONFIG["model_id"],
            "weights_sha256": _ensemble_weights_sha256(gnn_model_paths),
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

                    _gnn_recommendation = load_gnn_model_weights()
                    if _gnn_recommendation:
                        logger.info(f"GNN model loaded successfully from {', '.join(gnn_model_paths)}")
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
