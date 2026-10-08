"""Imported by backbones.build_dinov2 before torch.hub.load: disables xFormers in the DINOv2 layers (torch SDPA
instead). Raises if the dinov2 layers were already imported in this process with xFormers enabled."""
import os
import sys

if "dinov2.layers.attention" in sys.modules and sys.modules["dinov2.layers.attention"].XFORMERS_AVAILABLE:
    raise RuntimeError("dinov2 layers already imported with xFormers enabled in this process")
os.environ["XFORMERS_DISABLED"] = "1"
