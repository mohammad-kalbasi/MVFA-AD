import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted
from utils import encode_text_with_prompt_ensemble
from prompt import REAL_NAME, ORGAN_ALIASES


CLIP_MEAN = (0.4814, 0.4578, 0.4082)
CLIP_STD = (0.2686, 0.2613, 0.2758)


def run_inference(
    image: torch.Tensor,
    organ: str,
    ckpt_dir: str = "ckpt/zero-shot",
    model_name: str = "ViT-L-14-336",
    pretrained: str = "openai",
    clip_ckpt_path: Optional[str] = None,
    features: List[int] = [6, 12, 18, 24],
    normalize: bool = False,
    img_size: int = 240,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run anomaly detection on an input tensor.

    Parameters
    ----------
    image: torch.Tensor
        Input tensor of shape (B, C, H, W) with values in [0,1].
    organ: str
        Organ name (simple alias like ``"brain"``). See ``prompt.ORGAN_ALIASES``
        for supported names.
    ckpt_dir: str
        Directory containing zero-shot checkpoints named ``{organ}.pth``.
    model_name, pretrained, features
        CLIP backbone configuration. ``clip_ckpt_path`` optionally overrides
        the default CLIP checkpoint location.
    normalize: bool
        Apply CLIP mean/std normalisation after resizing.
    img_size: int
        Spatial size expected by the model (default 240).
    device: str
        Device to perform inference on.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(image_score, anomaly_map)`` where ``image_score`` is shape (B,) and
        ``anomaly_map`` has shape (B, img_size, img_size).
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    if image.shape[-2:] != (img_size, img_size):
        image = F.interpolate(
            image, size=(img_size, img_size), mode="bilinear", align_corners=False
        )

    if normalize:
        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        image = (image - mean) / std

    if clip_ckpt_path is not None:
        from CLIP import clip as clip_module

        clip_module._MODEL_CKPT_PATHS[model_name] = Path(clip_ckpt_path)

    clip_model = create_model(
        model_name=model_name,
        img_size=img_size,
        device=device,
        pretrained=pretrained,
        require_pretrained=True,
    ).eval()
    model = CLIP_Inplanted(clip_model=clip_model, features=features).to(device).eval()

    organ_key = ORGAN_ALIASES.get(organ.lower(), organ)
    ckpt = torch.load(os.path.join(ckpt_dir, f"{organ_key}.pth"), map_location=device)
    model.seg_adapters.load_state_dict(ckpt["seg_adapters"])
    model.det_adapters.load_state_dict(ckpt["det_adapters"])

    text_feat = encode_text_with_prompt_ensemble(
        clip_model, REAL_NAME[organ_key], device
    )

    with torch.no_grad(), torch.cuda.amp.autocast():
        _, seg_tokens, det_tokens = model(image)
        det_tokens = [t[:, 1:, :] for t in det_tokens]
        seg_tokens = [t[:, 1:, :] for t in seg_tokens]

        image_score = image.new_zeros(image.size(0))
        for t in det_tokens:
            t = t / t.norm(dim=-1, keepdim=True)
            score = 100.0 * t @ text_feat
            score = torch.softmax(score, dim=-1)[:, :, 1]
            image_score += score.mean(dim=-1)

        anomaly_maps = []
        for t in seg_tokens:
            t = t / t.norm(dim=-1, keepdim=True)
            amap = 100.0 * t @ text_feat
            B, L, C = amap.shape
            H = int(np.sqrt(L))
            amap = F.interpolate(
                amap.permute(0, 2, 1).view(B, 2, H, H),
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=True,
            )
            amap = torch.softmax(amap, dim=1)[:, 1]
            anomaly_maps.append(amap)
        anomaly_map = torch.stack(anomaly_maps).sum(dim=0)

    return image_score, anomaly_map


if __name__ == "__main__":
    # Example usage
    dummy = torch.randn(1, 3, 512, 512)
    score, amap = run_inference(dummy, "liver")
    print("score", score)
    print("anomaly map shape", amap.shape)

