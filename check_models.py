"""Quick check: instantiate our three models and show parameter counts."""
import torch
from src.models import make_model, count_parameters

for version in ("v1", "v2", "v3"):
    model = make_model(version)
    count = count_parameters(model)
    with torch.no_grad():
        shape = tuple(model(torch.randn(1, 3, 32, 32)).shape)
    print(f"{version}: {count:,} trainable parameters; output shape {shape}")
    assert count <= 5_000_000
    assert shape == (1, 10)
