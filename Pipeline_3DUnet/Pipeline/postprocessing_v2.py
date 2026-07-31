from __future__ import annotations

import ast
import os
import multiprocessing as mp
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
from skimage import measure as ski_measure


# -----------------------------------------------------------------------------
# CPU threading control (important when using multiprocessing)
# -----------------------------------------------------------------------------
# Set these early (at import) to prevent oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# -----------------------------------------------------------------------------
# Fast 8-neighborhood computation (vectorized, no Python loops)
# -----------------------------------------------------------------------------
_OFFSETS_8 = np.array(
    [[-1, 0], [-1, 1], [-1, -1],
     [ 1, 0], [ 1, 1], [ 1, -1],
     [ 0, -1], [ 0, 1]],
    dtype=np.int16
)


def _as_structured_rc(a: np.ndarray) -> np.ndarray:
    """
    View (N,2) int array as structured (N,) so np.unique / setdiff1d is vectorized.
    """
    a = np.ascontiguousarray(a)
    return a.view(np.dtype([("r", a.dtype), ("c", a.dtype)])).reshape(-1)


def get_surrounding_fast(coords_rc: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    coords_rc: (N,2) array of [row, col]
    returns: unique neighbor coords within bounds and not in coords_rc
    """
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


# -----------------------------------------------------------------------------
# I/O + config helpers
# -----------------------------------------------------------------------------
ArrayLike3D = Union[np.ndarray, torch.Tensor, str]


def load_classes_pixels(dict_loc: str) -> Dict[int, Tuple[int, int]]:
    """
    Loads the classesPixels dict from file (ast.literal_eval).
    Expected format: {class_ind: (pixel_minimum, connectivity), ...}
    """
    with open(dict_loc, "r") as f:
        return ast.literal_eval(f.read())


def read_segmentation(path: str) -> np.ndarray:
    """Read a segmentation image to numpy array (F,H,W)."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def _to_numpy_3d(x: ArrayLike3D) -> np.ndarray:
    """Convert torch / numpy / file path into (F,H,W) numpy array."""
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


# -----------------------------------------------------------------------------
# Single-process postprocessing core
# -----------------------------------------------------------------------------
def pixel_post_one_fast(
    seg: ArrayLike3D,
    classesPixels: Mapping[int, Tuple[int, int]],
    comb_thr: int,
    num_classes: int,
) -> np.ndarray:
    """
    seg: torch.Tensor / np.ndarray / filepath, shape (F,H,W) or (H,W)
    returns: np.ndarray (F,H,W)
    """
    arr = _to_numpy_3d(seg)

    if comb_thr == 1:
        arr = arr.copy()
        arr[arr == 9] = 10

    out = arr.copy()
    F, H, W = out.shape

    for frame in range(F):
        fr = out[frame]  # view (H,W)
        for class_ind, (pixel_minimum, connectivity) in classesPixels.items():
            mask = (fr == class_ind)

            labeled, count = ski_measure.label(mask, connectivity=connectivity, return_num=True)
            if count == 0:
                continue

            for region in ski_measure.regionprops(labeled):
                if region.area > pixel_minimum:
                    continue

                coords = region.coords  # (N,2)
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


# -----------------------------------------------------------------------------
# Multiprocessing across frames (Linux)
# -----------------------------------------------------------------------------
_MP_CLASSES: Mapping[int, Tuple[int, int]] | None = None
_MP_NUM_CLASSES: int | None = None


def _mp_init(classesPixels: Mapping[int, Tuple[int, int]], num_classes: int) -> None:
    global _MP_CLASSES, _MP_NUM_CLASSES
    _MP_CLASSES = classesPixels
    _MP_NUM_CLASSES = num_classes


def _mp_process_frame(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    i, fr2d = args
    # Process single frame by calling the same core on a 1-frame stack
    out_fr = pixel_post_one_fast(
        fr2d[None, ...],
        classesPixels=_MP_CLASSES,     # type: ignore[arg-type]
        comb_thr=0,
        num_classes=_MP_NUM_CLASSES,   # type: ignore[arg-type]
    )[0]
    return i, out_fr


def pixel_post_parallel_linux(
    seg: ArrayLike3D,
    classesPixels: Mapping[int, Tuple[int, int]],
    num_classes: int,
    *,
    comb_thr: int = 0,
    n_procs: int | None = None,
    chunksize: int = 2,
) -> np.ndarray:
    """
    Parallelize across frames. Linux only (uses fork).
    seg: torch / numpy / filepath. Returns np.ndarray (F,H,W).
    """
    arr = _to_numpy_3d(seg)

    # Apply comb_thr once (cheaper than in workers)
    if comb_thr == 1:
        arr = arr.copy()
        arr[arr == 9] = 10

    arr = np.ascontiguousarray(arr)
    F = arr.shape[0]

    print(f'Number of processes for pixel postprocessing: {n_procs}')

    if F <= 1 or (n_procs is not None and n_procs <= 1):
        return pixel_post_one_fast(arr, classesPixels, comb_thr=0, num_classes=num_classes)

    out = np.empty_like(arr)

    
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_procs,
        initializer=_mp_init,
        initargs=(classesPixels, num_classes),
    ) as pool:
        it = ((i, arr[i]) for i in range(F))
        for i, fr_out in pool.imap_unordered(_mp_process_frame, it, chunksize=chunksize):
            out[i] = fr_out

    return out


# -----------------------------------------------------------------------------
# Convenience wrapper (single entry point)
# -----------------------------------------------------------------------------
def pixel_postprocess(
    seg: ArrayLike3D,
    classesPixels: Mapping[int, Tuple[int, int]],
    *,
    num_classes: int,
    comb_thr: int = 0,
    n_procs: int | None = None,
    chunksize: int = 2,
) -> np.ndarray:
    """
    Unified entry point. If n_procs is None or <=1 -> single process.
    Otherwise -> multiprocessing across frames (Linux fork).
    """
    if n_procs is None or n_procs <= 1:
        return pixel_post_one_fast(seg, classesPixels, comb_thr=comb_thr, num_classes=num_classes)
    return pixel_post_parallel_linux(
        seg,
        classesPixels=classesPixels,
        num_classes=num_classes,
        comb_thr=comb_thr,
        n_procs=n_procs,
        chunksize=chunksize,
    )


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
# from time import time
# def main():
#     classes = load_classes_pixels(r'W:\rubenvdw\nnunetv2\nnUNet\nnunetv2\Data_info\Pixels_postprocessing.txt')
#     pred = read_segmentation(r'C:\Users\z923198\Documents\NLD-ISALA-0090_predictions_nopostproc.nii.gz')
#     t0 = time()
#     pred2 = pixel_postprocess(pred, classes, num_classes=14, comb_thr=0, n_procs=8, chunksize=2)
#     print("seconds:", time() - t0)

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()  # needed for spawn on Windows
#     main()
