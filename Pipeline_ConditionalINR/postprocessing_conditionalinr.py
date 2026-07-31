"""
Small-region cleanup for Conditional INR predictions.

Same technique as OCT_AID_lite_pipeline_local/postprocessing_v2.py (reused, not
duplicated logic drift): per frame, per class, find connected components of that
class; any component smaller than a configured pixel threshold gets reclassified
to whichever class dominates its immediate (8-connected) neighbourhood — or to
Background if that neighbourhood is >75% Background.

Left out on purpose vs the reference: the reference's `comb_thr` option hard-
merges its class 9 into class 10, specific to that pipeline's own class
taxonomy. Conditional INR's classes are merged upstream already, at label-
mapping time (dataset.py's _DEFAULT_LABEL_MAPPING) -- there is no equivalent
merge to carry over here without a fresh, project-specific decision.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import os
from typing import Dict, Mapping, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
from skimage import measure as ski_measure

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


_OFFSETS_8 = np.array(
    [[-1, 0], [-1, 1], [-1, -1],
     [ 1, 0], [ 1, 1], [ 1, -1],
     [ 0, -1], [ 0, 1]],
    dtype=np.int16
)


def _as_structured_rc(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype([("r", a.dtype), ("c", a.dtype)])).reshape(-1)


def get_surrounding_fast(coords_rc: np.ndarray, H: int, W: int) -> np.ndarray:
    """coords_rc: (N,2) [row,col]. Returns unique in-bounds neighbours not in coords_rc."""
    neigh = (coords_rc[:, None, :] + _OFFSETS_8[None, :, :]).reshape(-1, 2)
    m = (
        (neigh[:, 0] >= 0) & (neigh[:, 0] < H) &
        (neigh[:, 1] >= 0) & (neigh[:, 1] < W)
    )
    neigh = neigh[m]
    if neigh.size == 0:
        return neigh

    neigh_s = np.unique(_as_structured_rc(neigh))
    coords_s = _as_structured_rc(coords_rc)
    out_s = np.setdiff1d(neigh_s, coords_s, assume_unique=False)
    if out_s.size == 0:
        return np.empty((0, 2), dtype=coords_rc.dtype)
    return out_s.view(coords_rc.dtype).reshape(-1, 2)


ArrayLike3D = Union[np.ndarray, torch.Tensor, str]


def load_classes_pixels(dict_loc: str) -> Dict[int, Tuple[int, int]]:
    """Load {class_ind: (pixel_minimum, connectivity)} via ast.literal_eval."""
    with open(dict_loc, "r") as f:
        return ast.literal_eval(f.read())


def read_segmentation(path: str) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def _to_numpy_3d(x: ArrayLike3D) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x
        if arr.dim() == 2:
            arr = arr.unsqueeze(0)
        return arr.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x[None, ...] if x.ndim == 2 else x
    if isinstance(x, str):
        return read_segmentation(x)
    raise TypeError(f"Unsupported input type: {type(x)}")


def pixel_post_one_fast(
    seg: ArrayLike3D,
    classesPixels: Mapping[int, Tuple[int, int]],
    num_classes: int,
) -> np.ndarray:
    """seg: (F,H,W) or (H,W). Returns cleaned (F,H,W) array."""
    arr = _to_numpy_3d(seg)
    out = arr.copy()
    F, H, W = out.shape

    for frame in range(F):
        fr = out[frame]
        for class_ind, (pixel_minimum, connectivity) in classesPixels.items():
            mask = (fr == class_ind)
            labeled, count = ski_measure.label(mask, connectivity=connectivity, return_num=True)
            if count == 0:
                continue

            for region in ski_measure.regionprops(labeled):
                if region.area > pixel_minimum:
                    continue

                coords = region.coords
                neigh = get_surrounding_fast(coords, H, W)
                if neigh.shape[0] == 0:
                    continue

                neigh_classes = fr[neigh[:, 0], neigh[:, 1]].astype(np.int64, copy=False)
                counts = np.bincount(neigh_classes, minlength=num_classes)

                total = int(counts.sum())
                if total > 0 and (counts[0] / total) > 0.75:
                    new_class = 0
                else:
                    new_class = int(np.argmax(counts[1:]) + 1)

                fr[coords[:, 0], coords[:, 1]] = new_class

    return out


_MP_CLASSES: Mapping[int, Tuple[int, int]] | None = None
_MP_NUM_CLASSES: int | None = None


def _mp_init(classesPixels: Mapping[int, Tuple[int, int]], num_classes: int) -> None:
    global _MP_CLASSES, _MP_NUM_CLASSES
    _MP_CLASSES = classesPixels
    _MP_NUM_CLASSES = num_classes


def _mp_process_frame(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    i, fr2d = args
    out_fr = pixel_post_one_fast(
        fr2d[None, ...], classesPixels=_MP_CLASSES, num_classes=_MP_NUM_CLASSES,   # type: ignore[arg-type]
    )[0]
    return i, out_fr


def pixel_post_parallel(
    seg: ArrayLike3D,
    classesPixels: Mapping[int, Tuple[int, int]],
    num_classes: int,
    *,
    n_procs: int | None = None,
    chunksize: int = 2,
) -> np.ndarray:
    """Parallelize across frames (spawn context -- safe on Windows and Linux)."""
    arr = np.ascontiguousarray(_to_numpy_3d(seg))
    F = arr.shape[0]

    if F <= 1 or (n_procs is not None and n_procs <= 1):
        return pixel_post_one_fast(arr, classesPixels, num_classes=num_classes)

    out = np.empty_like(arr)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_procs, initializer=_mp_init, initargs=(classesPixels, num_classes)) as pool:
        it = ((i, arr[i]) for i in range(F))
        for i, fr_out in pool.imap_unordered(_mp_process_frame, it, chunksize=chunksize):
            out[i] = fr_out
    return out


def pixel_postprocess(
    seg: ArrayLike3D,
    classesPixels: Mapping[int, Tuple[int, int]],
    *,
    num_classes: int,
    n_procs: int | None = None,
    chunksize: int = 2,
) -> np.ndarray:
    """Single entry point. n_procs None/<=1 -> single process, else parallel across frames."""
    if n_procs is None or n_procs <= 1:
        return pixel_post_one_fast(seg, classesPixels, num_classes=num_classes)
    return pixel_post_parallel(seg, classesPixels, num_classes=num_classes, n_procs=n_procs, chunksize=chunksize)


def postprocess_nifti(
    input_path: str,
    output_path: str,
    classesPixels: Mapping[int, Tuple[int, int]],
    *,
    num_classes: int,
    n_procs: int | None = None,
) -> None:
    """Read a prediction .nii.gz, clean it, write the result -- preserves spacing/origin/direction."""
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)
    cleaned = pixel_postprocess(arr, classesPixels, num_classes=num_classes, n_procs=n_procs)

    out_img = sitk.GetImageFromArray(cleaned.astype(np.int16))
    out_img.CopyInformation(img)   # keep spacing/origin/direction identical to the raw prediction
    sitk.WriteImage(out_img, output_path)
