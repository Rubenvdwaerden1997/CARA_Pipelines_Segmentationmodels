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
from Pipeline_nnunetv2.Pipeline.predict import predict_from_files
from Pipeline_nnunetv2.Pipeline.postprocessing_v2 import pixel_postprocess


default_num_processes = 8 

def apply_connected_component_analysis(dict_loc):
    with open(dict_loc) as f:
        myData = f.read()
    classesPixels = ast.literal_eval(myData)

    return classesPixels


def preprocess_data(args, full_input_filename, LUT):
    file_name = full_input_filename.split('\\')[-1]

    # Remove the .dcm extension
    pullback_name = file_name.replace('.dcm', '')

    print('Reading pullback: ', pullback_name)
    #Load the input files to create a list of slices
    series_inputdata = pydicom.dcmread(
        full_input_filename,
        defer_size='512 KB',  # Skip file size calculation
        force=False           # Skip unnecessary conversions
    )
    series_array_inputdata = series_inputdata.pixel_array
    if len(series_array_inputdata.shape) == 3:
        print('Input data is 3D, but should be 4D (frames, height, width, channels)')
        print(f'File {full_input_filename} is not processed, please check the input shape! \n')
        return None
    if series_array_inputdata.shape[1] != 704 or series_array_inputdata.shape[2] != 704:
        print(f'Input data is not in the right shape, should be (frames, 704, 704, channels) but is {series_array_inputdata.shape}')
        print(f'File {full_input_filename} is not processed, please check the input shape! \n')
        return None
    
    inverted_circular_mask_dcm = np.invert(circular_mask_dcm)
    series_array_inputdata *= inverted_circular_mask_dcm
    #masked_series_array_inputdata_gray = rgb_to_grayscale_with_mapping(masked_series_array_inputdata)
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

def predict(frames_array_input_image, output_folder_file, k, predictor_reinitiate, postprocessing_connected_component_dict=None, predictor_used=None, num_processes=default_num_processes, batch_size=1, save_intermediate=False, save_intermediate_path=None, save_intermediate_name=None, save_intermediate_suffix=None, save_intermediate_compression=False, save_intermediate_compression_level=0, save_intermediate_compression_type='gzip', save_intermediate_compression_options=None, save_intermediate_compression_options_dict=None, save_intermediate_compression_options_dict_key=None, save_intermediate_compression_options_dict_value=None):
    # Placeholder for the prediction function
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
    
    time_start = time.time()
    prediction_results, predictor_used = predict_from_files(reshaped_array, predictor_reinitiate)
    time_end = time.time()
    print(f'Prediction time: {time_end - time_start} seconds')

    if postprocessing_connected_component_dict is not None:
        classesPixels = apply_connected_component_analysis(postprocessing_connected_component_dict)
        time_start = time.time()
        # Postprocess the prediction results
        prediction_results_postprocessed = pixel_postprocess(prediction_results, classesPixels, num_classes=14, comb_thr=0, n_procs=4)
        time_end = time.time()
        print(f'Postprocessing time: {time_end - time_start} seconds')
        prediction_image = sitk.GetImageFromArray(prediction_results_postprocessed)
    else:
        prediction_image = sitk.GetImageFromArray(prediction_results)

    # Save the image as a .nii.gz file
    sitk.WriteImage(prediction_image, output_folder_file)

    print(f"Prediction results saved to {output_folder_file}!")

    return predictor_used


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Script for preprocessing and prediction")
        parser.add_argument('--data_input', type=str, default=r'C:\Users\z923198\Documents\CFT', help="Path to input folder.")
        parser.add_argument('--data_output', type=str, default=r'C:\Users\z923198\Documents\CFT', help="Path to output folder.")
        parser.add_argument('--k', type=int, default=3, help="Number of frames to sample around.")
        parser.add_argument('--radius', type=int, default=352, help="Radius for processing.")
        parser.add_argument('--preprocessing', action='store_true', help="Enable preprocessing.")
        parser.add_argument('--predict', action='store_true', help="Enable prediction.")
        parser.add_argument("--postprocessing_connected_component_dict", type=str, default=r"W:\rubenvdw\Rick_externalvalidation20062025\Info_files\Pixels_postprocessing_7mm.txt")
        parser.add_argument('-model_results_folder', type=str, required=True, default=r'W:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_results')
        parser.add_argument('-d', type=str, required=True, default='Dataset701_Rickexternalvalidation', help='Dataset complete name')

        # Parse arguments
        args = parser.parse_args() #argv should be inside this and delete following lines
        LUT = build_lut(map_rgb_to_gray(), default_gray=0)
        circular_mask_dcm = create_circular_mask(704, 704, channels=3)

        input_folder = args.data_input
        output_folder = args.data_output
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Output folder '{output_folder}' created.")
        dcm_files = [f for f in os.listdir(input_folder) if f.endswith('.dcm')]

        if not dcm_files:
            print("No .dcm files found in the specified folder.")

        print(f"Found {len(dcm_files)} .dcm files. Starting processing...")

        # Set predictor to None, it will be initialized in the first iteration
        predictor_reinitiate = None
        # Run preprocessing if flag is set
        for dcm_file in dcm_files:
            time_start_loop = time.time()
            full_file_path = os.path.join(input_folder, dcm_file)
            output_file_path = os.path.join(output_folder, dcm_file.replace('.dcm', '_predictions.nii.gz'))
            print(f"Processing: {full_file_path}")
            if os.path.exists(output_file_path):
                print(f"File {output_file_path} already exists, skipping processing.")
                continue  # Skip processing if the file already exists
            # Run preprocessing if flag is set
            if args.preprocessing:
                time_start = time.time()
                frames_array = preprocess_data(args, full_file_path, LUT)
                time_end = time.time()
                if frames_array is None:
                    print(f"Skipping prediction for {dcm_file} due to preprocessing failure.")
                    continue
                else:
                    print(f"Preprocessing time: {time_end - time_start} seconds")
                    print("Preprocessing complete, frames processed.")

            # Run prediction if flag is set
            if args.predict and frames_array is not None:
                time_start = time.time()
                predictor_reinitiate = predict(frames_array, output_file_path, args.k, predictor_reinitiate, args.postprocessing_connected_component_dict)
                time_end = time.time()
                print(f"Prediction time: {time_end - time_start} seconds")
                print("Prediction complete, results saved.")
            else:
                print(f"Skipping prediction for {dcm_file} as preprocessing is not enabled or frames_array is None.")
            time_end_loop = time.time()
            print(f"Total processing time for {dcm_file}: {time_end_loop - time_start_loop} seconds")
            print("###########################################\n")
            
    except SystemExit as e:
        print(f"Error: {e}")
        sys.exit(2)  # Exit with error code 2