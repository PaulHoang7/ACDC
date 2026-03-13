from .unetr34 import UNetR34
from .attunetr34 import AttUNetR34
from .boundary_dsunetr34 import BoundaryDSUNetR34

MODEL_REGISTRY = {
    "m1_unetr34": UNetR34,
    "m2_attunetr34": AttUNetR34,
    "m3_boundarydsunetr34": BoundaryDSUNetR34,
}


def build_model(model_name: str, **kwargs):
    """Build a model by name from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
