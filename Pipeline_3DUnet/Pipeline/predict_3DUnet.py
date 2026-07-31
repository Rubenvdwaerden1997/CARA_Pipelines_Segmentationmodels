import os
import sys
import time
import numpy as np
import torch
import SimpleITK as sitk
from monai.inferers import sliding_window_inference

default_num_processes = 8

def load_model(model_architecture, model_path, device, checkpoint='best'):
    """
    Load the model from the checkpoint. Please add the model every time manually, which one you want to use

    Args:
        model_architecture (torch.nn.Module): Model architecture to use.
        model_base_path (str): Path to the model base.
        device (torch.device): Device to load the model on (CPU or GPU).
        checkpoint (str): Checkpoint file stem (without .pth) inside model_path.

    Returns:
        model: Loaded model.
    """

    model = model_architecture.to(device)
    checkpoint_path = os.path.join(model_path, f"{checkpoint}.pth")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    return model

def define_model_architecture(model_name, device):
    if model_name.lower() == "3dunet_jasper":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from Model.Z16_3D_includingpseudo import model_intialisation
        import monai

        model_args = {
            'spatial_dims': model_intialisation.spatial_dims,
            'in_channels': model_intialisation.in_channels,
            'out_channels': model_intialisation.out_channels,
            'channels': model_intialisation.channels,
            'strides': model_intialisation.strides,
            'num_res_units': model_intialisation.res_units,
            'dropout': model_intialisation.dropout,
            'kernel_size': model_intialisation.kernel_size,
        }
        model = monai.networks.nets.UNet(
            spatial_dims=model_args['spatial_dims'],
            in_channels=model_args['in_channels'],
            out_channels=model_args['out_channels'],
            channels=model_args['channels'],
            strides=model_args['strides'],
            num_res_units=model_args['num_res_units'],
            dropout=model_args['dropout'],
            kernel_size=model_args['kernel_size'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


    return model

def predict_from_files_3Dmodel(
    input_array,
    model_path,
    model_name,
    model_initiator=None,
    batch_size=1,
    patch_depth=16,
    overlap=0.0,
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
    # input_array is (N frames, H, W) or (N frames, H, W, C) -> reshape to (C, H, W, N).
    # Depth (frames) must be the LAST spatial dim here: the configured `strides`
    # downsample dims 0/1 (H, W) by 2**6=64 and dim 2 by only 4, which only lines
    # up with H=W=704 (704/64=11) and the 16-frame patch (16/4=4) if depth is last.
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, N), C=1
    elif input_tensor.ndim == 4:
        input_tensor = input_tensor.permute(3, 1, 2, 0)  # (C, H, W, N)
    else:
        raise ValueError(f"Unexpected input shape: {tuple(input_tensor.shape)}")
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, C, H, W, N)
    _, C, H, W, N = input_tensor.shape

    # ------------------------
    # Sliding-window inference along the frame axis.
    # roi_size matches H/W exactly, so windowing only happens along depth (N).
    # overlap=0.0 tiles the volume with no overlap (last window still gets
    # pulled back flush with the end by MONAI so it isn't padded/dropped);
    # overlap>0 blends neighboring windows to soften seams at the cost of
    # extra forward passes.
    # ------------------------
    start_infer_time = time.time()
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=(H, W, patch_depth),
            sw_batch_size=batch_size,
            predictor=model_initiator,
            overlap=overlap,
            mode="gaussian" if overlap > 0 else "constant",
        )
        preds = torch.argmax(logits, dim=1)  # (1, H, W, N)
    inference_time = time.time() - start_infer_time

    prediction_array = np.moveaxis(preds[0].cpu().numpy().astype(np.uint8), -1, 0)  # (N, H, W)
    print(f'Inference time: {inference_time} seconds')
    print(f'Prediction completed. Shape: {prediction_array.shape}')

    return prediction_array, model_initiator