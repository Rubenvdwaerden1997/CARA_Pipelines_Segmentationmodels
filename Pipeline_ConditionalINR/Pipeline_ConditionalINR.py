"""
Batch inference pipeline for the Conditional INR IV-OCT segmentation model.

Mirrors the contract of Pipeline_OCTAIDlite/Pipeline/Pipeline_withoutinternalsaving.py:
point it at a folder of raw .dcm pullbacks and it preprocesses (circular mask +
grayscale + normalize) and predicts (.nii.gz per pullback), skipping files whose
output already exists.

Usage:
    python Pipeline_ConditionalINR.py \
        --data_input  /path/to/dicom_folder \
        --data_output /path/to/output_folder \
        --preprocessing --predict \
        --model_dir   /path/to/saved_models_3D_conditional/<experiment_name> \
        --checkpoint  best \
        --overlap     0.5 \
        --device      cuda
"""

import argparse
import os
import sys
import time

import numpy as np
import pydicom

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, current_dir)

from conversion_utils import create_circular_mask, build_lut, map_rgb_to_gray, rgb_to_gray_lut
from conditional_inr_inference import load_conditional_inr_model, predict_conditionalinr
from postprocessing_conditionalinr import load_classes_pixels, postprocess_nifti

_DEFAULT_POSTPROCESS_CONFIG = os.path.join(current_dir, "postprocessing_classes_conditionalinr.txt")


def preprocess_dicom_volume(dcm_path, LUT, circular_mask_channel3, circular_mask_channel1):
    """DICOM -> preprocessed [D,704,704] float32 volume.

    Mirrors Pipeline_withoutinternalsaving.preprocess_data(): circular mask ->
    (RGB only) LUT grayscale conversion -> whole-volume z-score normalization.
    Deliberately DROPS the old pipeline's sliding_window_view 7-frame-context
    step -- Conditional INR consumes the full volume via 3D patches, not a
    fixed-channel 2D stack.

    Returns None (fail-soft, same as the old pipeline) if the DICOM shape
    doesn't match the expected (frames, 704, 704, channels).
    """
    series = pydicom.dcmread(dcm_path, defer_size='512 KB', force=False)
    arr = series.pixel_array

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=-1)

    if arr.shape[1] != 704 or arr.shape[2] != 704:
        print(f"Input data is not in the right shape, should be (frames, 704, 704, channels) "
              f"but is {arr.shape}")
        print(f"File {dcm_path} is not processed, please check the input shape!\n")
        return None

    if arr.shape[3] == 3:
        inv_mask = np.invert(circular_mask_channel3)
        arr_masked = arr * inv_mask
        gray = rgb_to_gray_lut(arr_masked, LUT)
    else:
        inv_mask = np.invert(circular_mask_channel1)
        arr_masked = arr * inv_mask
        gray = arr_masked.squeeze()

    gray = gray.astype(np.float32, copy=False)
    mean = gray.mean()
    std  = gray.std()
    gray = (gray - mean) / (std + 1e-8)

    return gray   # [D, 704, 704] float32


def main():
    parser = argparse.ArgumentParser(description="Conditional INR batch inference pipeline")
    parser.add_argument('--data_input',  type=str, required=True, help="Folder of raw .dcm pullback files")
    parser.add_argument('--data_output', type=str, required=True, help="Folder to write .nii.gz predictions to")
    parser.add_argument('--preprocessing', action='store_true', help="Enable DICOM preprocessing")
    parser.add_argument('--predict',       action='store_true', help="Enable prediction")
    parser.add_argument('--model_dir',   type=str, required=True,
                         help="Folder with best_model.pt/latest.pt + its yaml config")
    parser.add_argument('--checkpoint',  type=str, default='best', choices=['best', 'latest'])
    parser.add_argument('--overlap',     type=float, default=0.5,
                         help="Sliding z-patch overlap fraction [0.0-1.0]. "
                              "0.0 = no overlap (fastest), 0.5 = 50%% overlap (default), "
                              "1.0 = stride of 1 frame (slowest, max smoothing).")
    parser.add_argument('--chunk_size',  type=int, default=65536,
                         help="Voxels per INR forward pass (default 65536, up from "
                              "predict_pullback's own 16384 default). Raise if VRAM allows "
                              "for fewer/larger GPU calls; lower on out-of-memory errors.")
    parser.add_argument('--use_amp',     action='store_true',
                         help="Enable encoder-only bfloat16 autocast (off by default). "
                              "The FiLM-SIREN decode loop always stays fp32 regardless "
                              "(too boundary-sensitive to run in reduced precision, "
                              "measured ~95%% boundary-pixel agreement vs fp32 even with "
                              "the decode loop excluded). ~23%% faster if enabled, at that "
                              "boundary-accuracy risk.")
    parser.add_argument('--device',      type=str, default='cuda', help="cuda or cpu")
    parser.add_argument('--postprocess', action='store_true',
                         help="Enable small-region cleanup on the raw prediction. Writes a "
                              "second file alongside the raw one, with '_postprocessed' added "
                              "before the .nii.gz extension -- the raw prediction is always kept.")
    parser.add_argument('--postprocess_config', type=str, default=_DEFAULT_POSTPROCESS_CONFIG,
                         help="Path to the {class_index: (pixel_minimum, connectivity)} text file "
                              "(default: postprocessing_classes_conditionalinr.txt next to this "
                              "script -- currently placeholder thresholds, review before relying "
                              "on the output).")
    parser.add_argument('--postprocess_n_procs', type=int, default=4,
                         help="Parallelize cleanup across frames with this many processes "
                              "(default 4, matching the reference OCT_AID_lite pipeline).")
    args = parser.parse_args()

    os.makedirs(args.data_output, exist_ok=True)
    model_name = os.path.basename(args.model_dir.rstrip('/'))
    output_folder = os.path.join(args.data_output, model_name)
    os.makedirs(output_folder, exist_ok=True)

    LUT = build_lut(map_rgb_to_gray(), default_gray=0)
    circular_mask_channel3 = create_circular_mask(704, 704, channels=3)
    circular_mask_channel1 = create_circular_mask(704, 704, channels=1)

    dcm_files = [f for f in os.listdir(args.data_input) if f.endswith('.dcm')]
    if not dcm_files:
        print("No .dcm files found in the specified folder.")
    print(f"Found {len(dcm_files)} .dcm files. Starting processing...")

    model = cfg = device = None
    if args.predict or args.postprocess:
        model, cfg, device = load_conditional_inr_model(args.model_dir, args.checkpoint, args.device)

    classes_pixels = None
    if args.postprocess:
        classes_pixels = load_classes_pixels(args.postprocess_config)
        print(f"[Postprocess] Loaded thresholds from {args.postprocess_config}")

    for dcm_file in dcm_files:
        time_start_loop = time.time()
        full_file_path = os.path.join(args.data_input, dcm_file)
        output_file_path = os.path.join(output_folder, dcm_file.replace('.dcm', f'_predictions_overlap{args.overlap}.nii.gz'))
        postproc_file_path = output_file_path.replace('.nii.gz', '_postprocessed.nii.gz')
        print(f"Processing: {full_file_path}")

        raw_exists = os.path.exists(output_file_path)
        need_postprocess = args.postprocess and not os.path.exists(postproc_file_path)

        if raw_exists and not need_postprocess:
            print(f"Output file '{output_file_path}' already exists. Skipping processing for this file.")
            print("###########################################\n")
            continue

        if not raw_exists:
            volume = None
            if args.preprocessing:
                time_start = time.time()
                volume = preprocess_dicom_volume(full_file_path, LUT, circular_mask_channel3, circular_mask_channel1)
                time_end = time.time()
                if volume is None:
                    print(f"Skipping prediction for {dcm_file} due to preprocessing failure.")
                    continue
                else:
                    print(f"Preprocessing time: {time_end - time_start:.2f} seconds")

            if args.predict and volume is not None:
                time_start = time.time()
                predict_conditionalinr(volume, model, cfg, device, output_file_path,
                                        overlap_frac=args.overlap, chunk_size=args.chunk_size,
                                        use_amp=args.use_amp,
                                        postprocess_classes=classes_pixels if need_postprocess else None,
                                        postprocess_output_path=postproc_file_path,
                                        postprocess_n_procs=args.postprocess_n_procs)
                time_end = time.time()
                print(f"Prediction (+ postprocessing) time: {time_end - time_start:.2f} seconds")
                print("Prediction complete, results saved.")
                need_postprocess = False   # already handled in-memory above, skip the fallback below
            else:
                print(f"Skipping prediction for {dcm_file} as preprocessing is not enabled or volume is None.")
        else:
            print(f"Raw prediction '{output_file_path}' already exists, reusing it for postprocessing.")

        if need_postprocess:
            # Only reached when the raw file already existed from a previous run (no in-memory
            # `pred` available) -- falls back to reading it from disk for this file only.
            if os.path.exists(output_file_path):
                time_start = time.time()
                postprocess_nifti(output_file_path, postproc_file_path, classes_pixels,
                                   num_classes=cfg.num_classes, n_procs=args.postprocess_n_procs)
                time_end = time.time()
                print(f"Postprocessing time: {time_end - time_start:.2f} seconds")
                print(f"Postprocessed prediction saved -> {postproc_file_path}")
            else:
                print(f"Skipping postprocessing for {dcm_file}: raw prediction was not produced.")

        time_end_loop = time.time()
        print(f"Total processing time for {dcm_file}: {time_end_loop - time_start_loop:.2f} seconds")
        print("###########################################\n")

    print("All files processed.")


if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        print(f"Error: {e}")
        sys.exit(2)
