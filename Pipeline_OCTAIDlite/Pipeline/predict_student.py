from Unet_MONAI import UNet
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import segmentation_models_pytorch as smp

default_num_processes = 8

def load_model(model_architecture, model_path, device):
    """
    Load the model from the checkpoint. Please add the model every time manually, which one you want to use

    Args:
        model_architecture (torch.nn.Module): Model architecture to use.
        model_base_path (str): Path to the model base.
        device (torch.device): Device to load the model on (CPU or GPU).

    Returns:
        model: Loaded model.
    """

    model = model_architecture.to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model

def define_model_architecture(model_name, device):

    if model_name == 'UNet':
        model_args = {
            'spatial_dims': 2,
            'in_channels': 7,
            'out_channels': 14,
            'channels': [16, 32, 64, 128, 256, 256, 256, 256],
            'strides': [1, 2, 2, 2, 2, 2, 2],
            'kernel_size': 3,
            'num_res_units': 2,
            'act': 'LeakyReLU',
            'norm': 'INSTANCE',
            'deep_supr_num': 0,
            'deep_supr_weights': None,
            'deep_supervision': False
            }

        model = UNet(
        spatial_dims=model_args['spatial_dims'],
        in_channels=model_args['in_channels'],
        out_channels=model_args['out_channels'],
        channels=model_args['channels'],
        strides=model_args['strides'],
        kernel_size=model_args['kernel_size'],
        num_res_units=model_args['num_res_units'],
        act = model_args['act'],
        norm = model_args['norm']
        ).to(device)
    elif model_name.lower().startswith("deeplabv3plus"):
        model_args = {
            'encoder_name': "mit_b1",        # backbone
            'encoder_depth': 5,
            'encoder_output_stride': 16,
            'decoder_channels': 256,
            'decoder_atrous_rates': [12, 24, 36],
            'decoder_aspp_separable': True,
            'decoder_aspp_dropout': 0.0,
            'upsampling': 4,
            'spatial_dims': 2,
            'in_channels': 7,
            'out_channels': 14
            }

        model = smp.DeepLabV3Plus(
            encoder_name=model_args["encoder_name"],           # backbone, e.g. "resnet18"
            encoder_weights=None,                              # no pretrained weights
            encoder_depth=model_args.get("encoder_depth", 5), # default to 5 if not specified
            encoder_output_stride=model_args.get("encoder_output_stride", 16),
            in_channels=model_args['in_channels'],            # input channels
            classes=model_args['out_channels'],               # output channels / number of classes
            decoder_channels=model_args.get("decoder_channels", 256),
            decoder_atrous_rates=model_args.get("decoder_atrous_rates", (12, 24, 36)),
            decoder_aspp_separable=model_args.get("decoder_aspp_separable", True),
            decoder_aspp_dropout=model_args.get("decoder_aspp_dropout", 0.0),
            upsampling=model_args.get("upsampling", 4),
            aux_params=model_args.get("aux_params", None),
        ).to(device)
    elif model_name.lower().startswith("segformer"):
        model_args = {
            'encoder_name': "mit_b1",      # the backbone
            'in_channels': 7,                # input channels
            'out_channels': 14,              # number of classes
            'index_to_predict': 3,
            'decoder_channels': 256,
            'decoder_use_batchnorm': True,
            'decoder_attention_type': 'scse',
            'upsampling': 4,
            }
        model = smp.Segformer(
            encoder_name=model_args["encoder_name"],   # e.g. "mit_b0"
            encoder_weights=None,                        # no pretrained weights
            in_channels=model_args['in_channels'],      # input channels
            classes=model_args['out_channels'],         # output channels / number of classes
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


    return model

def predict_from_files_student_v2(
    input_array,
    model_path,
    model_name,
    model_initiator=None,
    batch_size=1,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ------------------------
    # Initialize model
    # ------------------------
    if model_initiator is None:
        model_architecture = define_model_architecture(model_name, device)
        print(f'Initializing from model path {model_path}, using device: {device}')
        start_init = time.time()
        model = load_model(model_architecture, model_path, device)
        init_time = time.time() - start_init
        print(f'Initialization time: {init_time:.2f}s')
        model_initiator = model.to(device)
    else:
        print(f'Using existing predictor using device: {device}')
        model_initiator = model_initiator.to(device)

    model_initiator.eval()

    # ------------------------
    # Prepare input
    # ------------------------
    # Convert to float tensor once
    input_tensor = torch.tensor(input_array, dtype=torch.float32)

    # Make sure input shape matches model
    # Expected shape: (batch, channels, H, W)
    if input_tensor.ndim == 4:  # (N, C, H, W)
        pass
    elif input_tensor.ndim == 5:  # e.g., (N, H, W, C)
        input_tensor = input_tensor.permute(0, 4, 1, 2)  # move channels to dim=1

    # ------------------------
    # Allocate output
    # ------------------------
    N, C, H, W = input_tensor.shape
    prediction_array = np.zeros((N, H, W), dtype=np.uint8)

    # ------------------------
    # Batch inference
    # ------------------------
    start_infer_time = time.time()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = input_tensor[start:end].to(device)

            logits = model_initiator(batch)
            # Take argmax directly on GPU to avoid CPU transfers
            preds = torch.argmax(logits,dim=1)  # shape (batch, H, W)

            # Transfer only the predicted labels to CPU
            prediction_array[start:end] = preds.cpu().numpy().astype(np.uint8)
    inference_time = time.time() - start_infer_time
    print(f'Inference time: {inference_time} seconds')
    print(f'Prediction completed. Shape: {prediction_array.shape}')

    return prediction_array, model_initiator
