import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
import time
import gc
import pydicom
import ast
from numpy.lib.stride_tricks import sliding_window_view
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '..', 'utils')
sys.path.insert(1, os.path.normpath(utils_path))
from conversion_utils import create_circular_mask, build_lut, map_rgb_to_gray, rgb_to_gray_lut
from Pipeline_nnunetv2.Pipeline.predict_v2 import predict_from_files
from Pipeline_nnunetv2.Pipeline.postprocessing_v2 import pixel_postprocess


default_num_processes = 8

def strip_pullback_extension(raw_id):
    """Strip a trailing .dcm/.nii.gz/.nii extension (case-insensitive) and
    surrounding whitespace from an Excel ID cell, e.g. 'NL-RUMC-00002-1-1.dcm'
    -> 'NL-RUMC-00002-1-1'."""
    s = str(raw_id).strip()
    for suffix in ('.nii.gz', '.dcm', '.nii'):
        if s.lower().endswith(suffix):
            return s[: -len(suffix)]
    return s


def find_nifti_files(root_dir):
    """Recursively map lowercased '<id>' (from '<id>.nii.gz') to the full
    path of that file under root_dir, so an Excel ID can be resolved to a
    file living in any subfolder."""
    mapping = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.nii.gz'):
                stem = fn[: -len('.nii.gz')]
                mapping[stem.lower()] = os.path.join(dirpath, fn)
    return mapping


def resolve_files_from_excel(excel_file, id_column, sheet_name, data_input):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in {excel_file}. Available columns: {list(df.columns)}")
    raw_ids = df[id_column].dropna().tolist()

    nifti_map = find_nifti_files(data_input)

    files_to_process = []
    missing = []
    for raw_id in raw_ids:
        base_id = strip_pullback_extension(raw_id)
        match = nifti_map.get(base_id.lower())
        if match is None:
            missing.append(base_id)
            continue
        files_to_process.append((match, base_id))

    print(f"Resolved {len(files_to_process)}/{len(raw_ids)} Excel ID(s) to a '<id>.nii.gz' file under {data_input}.")
    if missing:
        preview = missing[:20]
        print(f"Warning: {len(missing)} ID(s) had no matching file (skipped): {preview}"
              f"{' ...' if len(missing) > len(preview) else ''}")
    return files_to_process


def apply_connected_component_analysis(dict_loc):
    with open(dict_loc) as f:
        myData = f.read()
    classesPixels = ast.literal_eval(myData)

    return classesPixels

def load_raw_array(full_input_filename):
    """Load a raw pullback as (frames, H, W, 3) uint8 RGB, from either a
    .dcm file (pydicom, deferred pixel-data read) or a .nii/.nii.gz file
    (SimpleITK, RGB-encoded the same way as the .dcm pullbacks)."""
    name = full_input_filename.lower()
    if name.endswith('.dcm'):
        series_inputdata = pydicom.dcmread(
            full_input_filename,
            defer_size='512 KB',  # Skip file size calculation
            force=False           # Skip unnecessary conversions
        )
        return series_inputdata.pixel_array
    if name.endswith('.nii.gz') or name.endswith('.nii'):
        ref_img = sitk.ReadImage(full_input_filename)
        return sitk.GetArrayFromImage(ref_img)
    raise ValueError(f"Unsupported pullback file type: {full_input_filename} (expected .dcm or .nii/.nii.gz)")


def preprocess_data(args, full_input_filename, LUT):
    pullback_name = os.path.basename(full_input_filename)

    print('Reading pullback: ', pullback_name)
    #Load the input files to create a list of slices
    series_array_inputdata = load_raw_array(full_input_filename)
    if len(series_array_inputdata.shape) == 3:
        print('Input data is 3D, but should be 4D (frames, height, width, channels)')
        print(f'File {full_input_filename} is not processed, please check the input shape! \n')
        return None
    if series_array_inputdata.shape[1] != 704 or series_array_inputdata.shape[2] != 704:
        print(f'Input data is not in the right shape, should be (frames, 704, 704, channels) but is {series_array_inputdata.shape}')
        print(f'File {full_input_filename} is not processed, please check the input shape! \n')
        return None

    series_array_inputdata = series_array_inputdata[..., :3].astype(np.uint8, copy=False)  # drop alpha if present, match dcm's uint8 RGB
    inverted_circular_mask_dcm = np.invert(circular_mask_dcm)
    series_array_inputdata = series_array_inputdata * inverted_circular_mask_dcm
    masked_series_array_inputdata_gray = rgb_to_gray_lut(series_array_inputdata, LUT)
    masked_series_array_inputdata_gray = masked_series_array_inputdata_gray.astype(np.float32, copy=False)
    mean = masked_series_array_inputdata_gray.mean()
    std = masked_series_array_inputdata_gray.std()
    masked_series_array_inputdata_gray = (masked_series_array_inputdata_gray - mean) / (std + 1e-8)
    num_frames, H, W = masked_series_array_inputdata_gray.shape
    k = args.k
    window = 2 * k + 1

    # Only cast if needed (avoids copy)
    if masked_series_array_inputdata_gray.dtype != np.float32:
        image = masked_series_array_inputdata_gray.astype(np.float32, copy=False)
    else:
        image = masked_series_array_inputdata_gray

    # Pad only along frame axis
    padded = np.pad(
        image,
        pad_width=((k, k), (0, 0), (0, 0)),
        mode="constant"
    )

    # Sliding window view (zero-copy, safe)
    frames_array_inputimage = sliding_window_view(
        padded,
        window_shape=(window,),
        axis=0
    )
    gc.collect()
    return frames_array_inputimage

def predict_teachermodel(frames_array_input_image, output_folder_file,
            k, predictor_reinitiate, postprocessing_connected_component_dict=None, predictor_used=None,
            ):
    print('Running predictions...')
    #Make sure that the input image is in the right shape (No. of frames, channels, height, width)
    if frames_array_input_image.shape[1] == 704 and frames_array_input_image.shape[2] == 704 and frames_array_input_image.shape[3] == (k*2+1):
        # Reshape to (375, 7, 704, 704) keeping the last two dimensions as 704 and 704
        reshaped_array = frames_array_input_image.transpose(0, 3, 1, 2)
    elif frames_array_input_image.shape[1] == 704 and frames_array_input_image.shape[2] == 704 and frames_array_input_image.shape[0] == (k*2+1):
        # Reshape to (375, 7, 704, 704) keeping the last two dimensions as 704 and 704
        reshaped_array = frames_array_input_image.transpose(3, 0, 1, 2)
    else:
        print(f'Input image might not be in correct shape. Should be e.g. (375, 7, 704, 704) but is in shape: {frames_array_input_image.shape}')
        print(f'File is not predicted, please check the input shape! Would have been saved as {output_folder_file} \n')
        return None

    prediction_results, predictor_used = predict_from_files(reshaped_array, predictor_reinitiate)

    if postprocessing_connected_component_dict is not None:
        print('Applying connected component analysis for postprocessing...')
        classesPixels = apply_connected_component_analysis(postprocessing_connected_component_dict)
        time_start = time.time()
        print(f'Shape before postprocessing: {prediction_results.shape}')
        # os.cpu counts the number of logical cores, which may be more than physical cores
        prediction_results_postprocessed = pixel_postprocess(prediction_results, classesPixels, num_classes=14, comb_thr=0, n_procs=4)
        time_end = time.time()
        print(f'Postprocessing time: {time_end - time_start} seconds')
        prediction_image = sitk.GetImageFromArray(prediction_results_postprocessed)
    else:
        print('No postprocessing applied as no dictionary provided')
        prediction_image = sitk.GetImageFromArray(prediction_results)

    # Save the image as a .nii.gz file
    sitk.WriteImage(prediction_image, output_folder_file)

    print(f"Prediction results saved to {output_folder_file}!")

    return predictor_used


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Script for preprocessing and prediction")
        parser.add_argument('--data_input', type=str, default=r'C:\Users\z923198\Documents\CFT', help="Path to input folder. In plain mode, .dcm files directly inside this folder are processed. With --excel_file, this folder (searched recursively, including subfolders) is instead searched for a '<id>.nii.gz' file per Excel row.")
        parser.add_argument('--data_output', type=str, default=r'C:\Users\z923198\Documents\CFT', help="Path to output folder.")
        parser.add_argument('--excel_file', type=str, default=None, help="Excel file listing pullback IDs to process (one per row). When given, --data_input is searched recursively for '<id>.nii.gz' per row instead of listing .dcm files directly in --data_input.")
        parser.add_argument('--id_column', type=str, default='DICOM_file', help="Column in --excel_file holding the pullback ID (a trailing .dcm/.nii/.nii.gz extension, if present, is stripped before searching for '<id>.nii.gz').")
        parser.add_argument('--sheet_name', type=str, default='0', help="Sheet name or 0-based index in --excel_file (default: first sheet).")
        parser.add_argument('--k', type=int, default=3, help="Number of frames to sample around.")
        parser.add_argument('--radius', type=int, default=352, help="Radius for processing.")
        parser.add_argument('--preprocessing', action='store_true', help="Enable preprocessing.")
        parser.add_argument('--predict', action='store_true', help="Enable prediction.")
        parser.add_argument("--postprocessing_connected_component_dict", type=str, default=r"W:\rubenvdw\Rick_externalvalidation20062025\Info_files\Pixels_postprocessing_7mm.txt")
        parser.add_argument('-model_results_folder', type=str, default=r'W:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_results')
        parser.add_argument('-d', type=str, default='Dataset701_Rickexternalvalidation', help='Dataset complete name')

        # Parse arguments
        args = parser.parse_args()
        os.makedirs(args.data_output, exist_ok=True)

        model_name = args.d
        input_folder = args.data_input
        output_folder_teacher = args.data_output
        # LUT can be created before deployment
        LUT = build_lut(map_rgb_to_gray(), default_gray=0)
        circular_mask_dcm = create_circular_mask(704, 704, channels=3)

        if not os.path.exists(output_folder_teacher):
            os.makedirs(output_folder_teacher)
            print(f"Output folder '{output_folder_teacher}' created.")
        if args.excel_file:
            sheet_name = int(args.sheet_name) if args.sheet_name.lstrip('-').isdigit() else args.sheet_name
            files_to_process = resolve_files_from_excel(args.excel_file, args.id_column, sheet_name, input_folder)
        else:
            pullback_files = sorted(f for f in os.listdir(input_folder)
                                    if f.lower().endswith(('.dcm', '.nii.gz', '.nii')))
            if not pullback_files:
                print("No .dcm/.nii.gz/.nii files found in the specified folder.")
            files_to_process = [(os.path.join(input_folder, f), strip_pullback_extension(f)) for f in pullback_files]

        print(f"Found {len(files_to_process)} file(s). Starting processing...")

        predictor_reinitiate_teacher = None
        for full_file_path, pullback_name in files_to_process:
            time_start_loop = time.time()
            output_file_path_teacher = os.path.join(output_folder_teacher, f'{pullback_name}_predictions.nii.gz')
            print(f"Processing: {full_file_path}")
            if os.path.exists(output_file_path_teacher):
                print(f"Output file '{output_file_path_teacher}' already exists. Skipping processing for this file.")
                print("###########################################\n")
                continue

            # Run preprocessing if flag is set
            if args.preprocessing:
                time_start = time.time()
                frames_array = preprocess_data(args, full_file_path, LUT)
                time_end = time.time()
                if frames_array is None:
                    print(f"Skipping prediction for {pullback_name} due to preprocessing failure.")
                    continue
                else:
                    print(f"Preprocessing time: {time_end - time_start} seconds")

            # Run prediction if flag is set
            if args.predict and frames_array is not None:
                time_start = time.time()
                predictor_reinitiate_teacher = predict_teachermodel(
                    frames_array, output_file_path_teacher, args.k, predictor_reinitiate_teacher,
                    postprocessing_connected_component_dict=args.postprocessing_connected_component_dict,
                    )
                time_end = time.time()
                print(f"Prediction + Postprocessing time: {time_end - time_start} seconds")
                print("Prediction complete, results saved.")
            else:
                print(f"Skipping prediction for {pullback_name} as preprocessing is not enabled or frames_array is None.")
            time_end_loop = time.time()
            print(f"Total processing time for teacher prediction {pullback_name}: {time_end_loop - time_start_loop} seconds")
            print("###########################################\n")
        print("All files processed.")

    except SystemExit as e:
        print(f"Error: {e}")
        sys.exit(2)  # Exit with error code 2
