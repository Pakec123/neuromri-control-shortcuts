"""Offline-only, hash-verified pretrained image encoders."""
import json
import io
import os
from pathlib import Path
import zipfile
import torch
from torch import nn
import yaml
from .common import digest


class BGRBackbone(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone=backbone

    def forward(self,x):
        return self.backbone(x[:,[2,1,0]]).flatten(1)


def encoder(name, weights_dir="weights/files", manifest="weights/manifest.yaml"):
    e = yaml.safe_load(Path(manifest).read_text())["encoders"][name]
    if e["status"] != "available":
        raise ValueError(f"{name}: {e['status']}")
    root = Path(weights_dir)
    for a in e["artifacts"]:
        if not (root / a["file"]).exists() or digest(root / a["file"]) != a["sha256"]:
            raise RuntimeError(f"Missing/corrupt {a['file']}; run neuromri.fetch_weights on login node")
    weight = root / e["artifacts"][0]["file"]
    mean, std = (.485,.456,.406), (.229,.224,.225)
    if e["backend"] == "torchvision":
        import torchvision.models as tv
        model = tv.get_model(e["architecture"], weights=None)
        state = torch.load(weight, map_location="cpu", weights_only=True)
        if name == "densenet121":
            import re
            state = {re.sub(r"(denselayer\d+\.(?:norm|relu|conv))\.([12]\.)", r"\1\2", k): v for k,v in state.items()}
        model.load_state_dict(state)
        if name.startswith("resnet"):
            dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "densenet121":
            dim = model.classifier.in_features
            model.classifier = nn.Identity()
        elif name in ["convnext_tiny", "efficientnet_v2_s"]:
            dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
        elif name == "vit_b_16":
            dim = model.heads.head.in_features
            model.heads = nn.Identity()
        elif name == "swin_t":
            dim = model.head.in_features
            model.head = nn.Identity()
        else:
            raise ValueError(name)
    elif e["backend"] == "dinov2":
        os.environ["XFORMERS_DISABLED"] = "1"
        from .fetch_weights import prepare_dino
        code = prepare_dino(root,e["revision"],verify_only=True)
        model = torch.hub.load(str(code), e["architecture"], source="local", pretrained=False)
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
        dim = model.embed_dim
        # The official 518px table otherwise invokes nondeterministic CUDA bicubic
        # backward at every 224px training step. Resize once on CPU, then learn
        # the resulting 16x16 position table directly. Determinism stays strict.
        with torch.no_grad():
            pos = model.interpolate_pos_encoding(torch.zeros(1,257,dim),224,224).clone()
        model.pos_embed = nn.Parameter(pos)
    elif e["backend"] == "radimagenet":
        import torchvision.models as tv
        import hashlib
        with zipfile.ZipFile(weight) as z:
            payload=z.read(e["member"])
        if hashlib.sha256(payload).hexdigest()!=e["weight_sha256"]:
            raise ValueError("RadImageNet member checksum mismatch")
        state=torch.load(io.BytesIO(payload),map_location="cpu",weights_only=True)
        base=tv.resnet50(weights=None)
        seq=nn.Sequential(*list(base.children())[:9])
        # Original state keys are prefixed by the example's Backbone.backbone.
        seq.load_state_dict({k.removeprefix("backbone."):v for k,v in state.items()})
        model=BGRBackbone(seq)
        dim=2048
        mean,std=(.5,.5,.5),(.5,.5,.5)
    elif e["backend"] == "open_clip_visual":
        from open_clip.timm_model import TimmModel
        cfg = json.loads((root / "biomedclip-open_clip_config.json").read_text())
        v = cfg["model_cfg"]["vision_cfg"]
        model = TimmModel(model_name=v["timm_model_name"], embed_dim=cfg["model_cfg"]["embed_dim"],
            image_size=v["image_size"], pool=v.get("timm_pool", "avg"),
            proj=v.get("timm_proj", "linear"), proj_bias=v.get("timm_proj_bias",False), pretrained=False)
        state = torch.load(weight, map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("visual."): t for k,t in state.items() if k.startswith("visual.")})
        dim = cfg["model_cfg"]["embed_dim"]
        mean, std = cfg["preprocess_cfg"]["mean"], cfg["preprocess_cfg"]["std"]
    else:
        raise ValueError(e["backend"])
    try:
        # Keep attention deterministic for timm-backed models too.
        from timm.layers import set_fused_attn
        set_fused_attn(False)
        for mod in model.modules():
            if hasattr(mod, "fused_attn"):
                mod.fused_attn = False
    except ImportError:
        pass
    return model, dim, mean, std
