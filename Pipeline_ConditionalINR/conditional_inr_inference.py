"""
Conditional INR model loading + inference + saving, playing the role
predict_student.py plays in the Pipeline_OCTAIDlite student-model pipeline.

Reuses the training/inference code in Conditional_INR/Training_model and the
already-working single-pullback helpers in
Conditional_INR/predict_singlepullback_conditional.py (imported via sys.path
rather than duplicated, to avoid drifting from actively-maintained training
code). This pipeline lives in a separate repo (CARA_Pipelines_Segmentationmodels,
moved here 2026-07-31 from inside Conditional_INR itself), so the Conditional_INR
repo root can no longer be derived from this file's own location -- it's an
explicit cross-repo path (dual local/cluster, same pattern used elsewhere in
this codebase, e.g. Teacher_student/Codes/Predict/Predict_Evaluate_dict_onefold.py's
nnunetv2 imports). Whichever path exists on the current machine resolves; the
other is a harmless no-op sys.path entry.
"""

import sys
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import yaml

_CONDITIONAL_INR_REPO_LOCAL   = Path("W:/rubenvdw/Conditional_INR")
_CONDITIONAL_INR_REPO_CLUSTER = Path("/data/diag/rubenvdw/Conditional_INR")
for _repo_root in (_CONDITIONAL_INR_REPO_LOCAL, _CONDITIONAL_INR_REPO_CLUSTER):
    sys.path.insert(0, str(_repo_root / "Training_model"))
    sys.path.insert(0, str(_repo_root))

from config import Config                                          # Training_model/config.py
from model import ConditionalINR, resize_volume                    # Training_model/model.py
from inference import predict_pullback                             # Training_model/inference.py
from predict_singlepullback_conditional import build_config_from_yaml, find_yaml
from postprocessing_conditionalinr import pixel_postprocess         # same folder


def load_conditional_inr_model(
    model_dir:  str,
    checkpoint: str = "best",
    device_str: str = "cuda",
):
    """Load yaml -> Config, instantiate ConditionalINR, load best_model.pt/latest.pt.

    Mirrors the Config/checkpoint-loading block of
    predict_singlepullback_conditional.py's main().
    """
    model_dir = Path(model_dir)
    yaml_path = find_yaml(model_dir)
    with open(yaml_path) as f:
        yml = yaml.safe_load(f)
    cfg = build_config_from_yaml(yml)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt_path = model_dir / ("best_model.pt" if checkpoint == "best" else "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = ConditionalINR(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[ConditionalINR] Loaded {ckpt_path.name} (epoch {ckpt.get('epoch', '?')}) "
          f"on {device}, config={yaml_path.name}")
    return model, cfg, device


@torch.no_grad()
def predict_conditionalinr(
    volume:            np.ndarray,   # [D, native_xy_size, native_xy_size] float32, z-score normalised
    model:              ConditionalINR,
    cfg:                Config,
    device:             torch.device,
    output_file_path:   str,
    overlap_frac:       float = 0.5,
    chunk_size:         int   = 65536,
    use_amp:            bool  = False,
    postprocess_classes:     Optional[Mapping[int, Tuple[int, int]]] = None,
    postprocess_output_path: Optional[str] = None,
    postprocess_n_procs:     Optional[int] = 4,
    eval_resolution:         str = "native",
) -> None:
    """Resize to cfg.resize_to (if set) for inference, run predict_pullback() querying
    the INR at native resolution, and save as .nii.gz with the correct NATIVE spacing
    (not the resized-grid spacing).

    eval_resolution: "native" (default, deployment behaviour) queries the INR at
    cfg.native_xy_size whenever cfg.resize_to differs from it, so the saved
    prediction is always full native resolution -- this is what every existing
    caller (the batch pipeline, and Metrics/Pullback_prediction.py's default run)
    gets, unchanged. "encoder" instead forces query_hw=None, so predict_pullback
    queries the INR on volume_infer's own (already-resized) grid -- the prediction
    comes back at cfg.resize_to (or native, if resize_to=0) instead. Use "encoder"
    to score the model in isolation from the native-query decoupling: how good are
    its own learned predictions at the resolution it was actually trained/validated
    on, before any native-resolution query-time upscaling is applied. Ground truth
    must be downsampled to match (nearest-neighbour, via model.resize_labels) by the
    caller before scoring -- this function only controls the resolution of `pred`.

    postprocess_classes: if given, small-region cleanup runs directly on the
    in-memory native-resolution `pred` array (no disk round-trip -- fastest
    path) and writes a second file at postprocess_output_path, alongside the
    always-written raw prediction. postprocess_n_procs mirrors the reference
    OCT_AID_lite pipeline's hardcoded default (4).

    chunk_size: voxels per INR forward pass (bumped from predict_pullback's own
    16384 default — fewer, larger GPU calls means less per-chunk overhead;
    raise further if VRAM allows, lower it if you hit an out-of-memory error).
    This is exact/lossless — just batching, not precision — safe to always leave up.

    use_amp: bfloat16 autocast, restricted to the encoder only (predict_pullback
    keeps the FiLM-SIREN decode loop in full fp32 regardless of this flag — see
    its docstring). Default False: measured locally that even encoder-only
    autocast drops boundary-pixel agreement to ~95% vs fp32 (vs ~100% in the
    interior) — omega_0=30 amplifies small upstream perturbations right where
    this model's decoder is already known to be least robust (the phantom-blob
    issue). Only ~23% faster in this restricted form (most of the cost is the
    always-fp32 decode loop), so the accuracy risk isn't worth it by default.
    Opt in explicitly if you want the extra speed and can tolerate that risk.
    """
    assert eval_resolution in ("native", "encoder"), \
        f"eval_resolution must be 'native' or 'encoder', got {eval_resolution!r}"
    volume = volume.astype(np.float32, copy=False)

    if cfg.resize_to:
        volume_infer = resize_volume(torch.from_numpy(volume), cfg.resize_to).numpy()
    else:
        volume_infer = volume

    D, H, W = volume_infer.shape
    # Query the INR at native resolution whenever the encoder input was downsized rather
    # than nearest-neighbor-upsampling the discrete (post-argmax) prediction afterward.
    # decode_3d/normalize_coords_3d decouple query-coordinate resolution from encoder-input
    # resolution (Training_model/model.py), so predict_pullback's query_hw genuinely
    # re-evaluates the continuous INR at every native pixel -- this used to instead resize
    # a coarse resize_to-resolution label mask up to native with resize_labels, which is a
    # cruder, blockier/staircased approximation of the same thing. query_hw=None (native ==
    # resize_to, or no resize at all) keeps predict_pullback's own default behaviour.
    #
    # eval_resolution="encoder" overrides this: query_hw stays None regardless of
    # resize_to, so predict_pullback queries on volume_infer's own (H, W) instead --
    # i.e. the model's own training/validation resolution, not native. Used to score
    # the model in isolation from the native-query decoupling (see docstring).
    if eval_resolution == "encoder":
        query_hw = None
    else:
        query_hw = cfg.native_xy_size if (cfg.resize_to and cfg.resize_to != cfg.native_xy_size) else None
    print(f"[ConditionalINR] Inference: D={D} H={H} W={W} overlap_frac={overlap_frac} "
          f"chunk_size={chunk_size} use_amp={use_amp} eval_resolution={eval_resolution} query_hw={query_hw}")
    pred = predict_pullback(model, volume_infer, cfg, device, chunk_size=chunk_size,
                             overlap_frac=overlap_frac, use_amp=use_amp,
                             query_hw=query_hw)  # [D, query_hw or H, query_hw or W] int64

    # out_xy_spacing is the correct physical pixel spacing for whatever grid `pred`
    # actually landed on above:
    #   - eval_resolution="encoder": pred is on volume_infer's own (resize_to, or
    #     native if resize_to=0) grid -- cfg.xy_spacing already IS that grid's
    #     spacing (Config.__post_init__ rescales it up from native when resize_to
    #     is set), no further conversion needed.
    #   - eval_resolution="native" (default): pred is native resolution -- undo
    #     Config.__post_init__'s upscaling to recover the true native spacing.
    if eval_resolution == "encoder":
        out_xy_spacing = cfg.xy_spacing
    elif cfg.resize_to and cfg.resize_to != cfg.native_xy_size:
        out_xy_spacing = cfg.xy_spacing * (cfg.resize_to / cfg.native_xy_size)
    else:
        out_xy_spacing = cfg.xy_spacing

    img = sitk.GetImageFromArray(pred.astype(np.int16))
    img.SetSpacing([out_xy_spacing, out_xy_spacing, cfg.z_spacing])
    sitk.WriteImage(img, str(output_file_path))
    print(f"Saved -> {output_file_path}")

    if postprocess_classes is not None:
        # Cleans the same in-memory `pred` array directly -- no re-read of the file
        # just written, unlike calling postprocess_nifti() on the saved path.
        cleaned = pixel_postprocess(pred, postprocess_classes, num_classes=cfg.num_classes,
                                     n_procs=postprocess_n_procs)
        post_img = sitk.GetImageFromArray(cleaned.astype(np.int16))
        post_img.SetSpacing([out_xy_spacing, out_xy_spacing, cfg.z_spacing])
        sitk.WriteImage(post_img, str(postprocess_output_path))
        print(f"Saved -> {postprocess_output_path}")
