import numpy as np
import SimpleITK as sitk
import cv2
from matplotlib.colors import ListedColormap
import base64
import os
import os.path as osp
from time import time
import numpy as np
import SimpleITK as sitk
from glob import glob
from numba import njit, prange

import einops
from networkx import out_degree_centrality
import numpy as np
import SimpleITK  as sitk

# -------------------- RGB->GRAY LUT utils --------------------
@njit(parallel=True, fastmath=True, cache=True)
def pack_rgb_numba(rgb_u8: np.ndarray) -> np.ndarray:
    """
    rgb_u8: (N, 3) uint8
    returns: (N,) uint32 packed as 0xRRGGBB
    """
    n = rgb_u8.shape[0]
    result = np.empty(n, dtype=np.uint32)
    for i in prange(n):
        result[i] = (np.uint32(rgb_u8[i, 0]) << 16) | (np.uint32(rgb_u8[i, 1]) << 8) | np.uint32(rgb_u8[i, 2])
    return result

@njit(parallel=True, fastmath=True, cache=True)
def lut_lookup_numba(packed: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply LUT lookup in parallel"""
    n = packed.shape[0]
    result = np.empty(n, dtype=np.uint8)
    for i in prange(n):
        result[i] = lut[packed[i]]
    return result


def create_circular_mask(h, w, center=None, radius=None, channels=3):
    """Apply the circular mask to remove the Abbott watermark

    Args:
        h (int): x dim in image
        w (int): y dim in image
        center (tuple, optional): coords of the center of the circular mask in the image. Defaults to None.
        radius (int, optional): radius of the created circular mask. Defaults to None.
        channels (int, optional): number of color channels in the image. Defaults to 3.

    Returns:
        np.array: Maskee image
    """

    if center is None: # use the middle of the image
        center = (int(w/2)-0.5, int(h/2)-0.5)

    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius+0.5
    if channels != 0:
        mask = np.expand_dims(mask, axis=(0,-1))  # Add an extra dimension for the color channels
        mask = np.repeat(mask, channels, axis=-1)  # Repeat the mask for each color channel

    return mask

def sample_around(image, frame, k):
    """Get k frames around specific frame with annotation

    Args:
        image (np.array): Frame with annotation
        frame (int): Nº frame we are looking
        k (int): Number of frame to sample before and after

    Returns:
        np.array: Volume that contains the frame with annotation in the middle and the k frames before and after
    """

    #Get neighbouring frakes
    neighbors = np.arange(frame-k, frame+k+1)

    frames_around = np.zeros((image.shape[1], image.shape[2], len(neighbors)))

    for i in range(len(neighbors)):

        #Case in which annotation is the first or last frame of the pullback (append black frames in that case)
        if neighbors[i] < 0 or neighbors[i] >= image.shape[0]:

            frames_around[:,:,i] = np.zeros((image.shape[1], image.shape[2]))

        else:
            frames_around[:,:,i] = image[neighbors[i],:,:]

    return frames_around

# -------------------- RGB->GRAY LUT utils --------------------

@njit(parallel=True, fastmath=True, cache=True)
def pack_rgb_numba(rgb_u8: np.ndarray) -> np.ndarray:
    """
    rgb_u8: (N, 3) uint8
    returns: (N,) uint32 packed as 0xRRGGBB
    """
    n = rgb_u8.shape[0]
    result = np.empty(n, dtype=np.uint32)
    for i in prange(n):
        result[i] = (np.uint32(rgb_u8[i, 0]) << 16) | (np.uint32(rgb_u8[i, 1]) << 8) | np.uint32(rgb_u8[i, 2])
    return result

@njit(parallel=True, fastmath=True, cache=True)
def lut_lookup_numba(packed: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply LUT lookup in parallel"""
    n = packed.shape[0]
    result = np.empty(n, dtype=np.uint8)
    for i in prange(n):
        result[i] = lut[packed[i]]
    return result


def map_rgb_to_gray() -> dict[tuple[int, int, int], int]:
    return {
        (0, 0, 0): 0,
        (5, 1, 0): 1,
        (10, 1, 0): 2,
        (14, 1, 0): 4,
        (17, 2, 0): 5,
        (19, 2, 0): 7,
        (21, 2, 0): 8,
        (23, 3, 0): 9,
        (26, 3, 0): 10,
        (27, 3, 0): 11,
        (29, 3, 0): 12,
        (30, 3, 0): 13,
        (32, 4, 0): 14,
        (33, 4, 0): 15,
        (36, 4, 0): 16,
        (38, 4, 0): 17,
        (40, 5, 0): 18,
        (42, 5, 0): 20,
        (43, 5, 0): 21,
        (45, 6, 0): 22,
        (47, 6, 0): 23,
        (48, 6, 0): 24,
        (51, 7, 1): 25,
        (53, 8, 1): 26,
        (54, 8, 1): 27,
        (56, 9, 1): 28,
        (58, 10, 1): 29,
        (59, 10, 1): 30,
        (61, 11, 1): 31,
        (63, 13, 1): 33,
        (65, 14, 1): 34,
        (67, 14, 1): 35,
        (68, 15, 1): 36,
        (70, 16, 1): 37,
        (71, 17, 1): 38,
        (73, 17, 1): 39,
        (74, 18, 1): 40,
        (76, 19, 1): 41,
        (79, 20, 1): 42,
        (80, 21, 1): 43,
        (81, 22, 1): 44,
        (83, 24, 1): 46,
        (85, 25, 1): 47,
        (86, 25, 1): 48,
        (88, 26, 1): 49,
        (90, 27, 1): 50,
        (92, 28, 2): 51,
        (94, 29, 2): 52,
        (95, 29, 2): 53,
        (97, 30, 2): 54,
        (98, 31, 2): 55,
        (99, 32, 2): 56,
        (101, 34, 2): 58,
        (103, 36, 2): 59,
        (104, 37, 2): 60,
        (107, 38, 2): 61,
        (108, 39, 2): 62,
        (109, 40, 2): 63,
        (111, 41, 2): 64,
        (112, 42, 2): 65,
        (114, 43, 2): 66,
        (115, 44, 2): 67,
        (117, 45, 2): 68,
        (119, 46, 2): 69,
        (121, 48, 2): 71,
        (123, 49, 2): 72,
        (124, 50, 2): 73,
        (125, 52, 2): 74,
        (127, 53, 2): 75,
        (129, 54, 2): 76,
        (130, 55, 3): 77,
        (131, 56, 3): 78,
        (132, 57, 3): 79,
        (134, 58, 3): 80,
        (136, 59, 3): 81,
        (137, 60, 3): 82,
        (139, 62, 4): 84,
        (140, 63, 4): 85,
        (141, 65, 4): 86,
        (143, 66, 4): 87,
        (144, 67, 4): 88,
        (145, 68, 4): 89,
        (146, 69, 4): 90,
        (148, 70, 5): 91,
        (150, 72, 5): 92,
        (151, 73, 5): 93,
        (152, 73, 5): 94,
        (154, 75, 5): 95,
        (156, 77, 6): 97,
        (157, 78, 6): 98,
        (158, 79, 6): 99,
        (159, 81, 6): 100,
        (161, 82, 6): 101,
        (163, 83, 7): 102,
        (164, 84, 8): 103,
        (165, 85, 8): 104,
        (166, 87, 8): 105,
        (167, 89, 9): 106,
        (168, 91, 9): 107,
        (169, 93, 9): 109,
        (170, 94, 10): 110,
        (171, 96, 10): 111,
        (171, 98, 10): 112,
        (172, 99, 11): 113,
        (173, 100, 11): 114,
        (174, 101, 11): 115,
        (175, 103, 12): 116,
        (175, 105, 12): 117,
        (177, 107, 12): 118,
        (178, 108, 13): 119,
        (179, 110, 13): 120,
        (180, 112, 13): 122,
        (181, 113, 14): 123,
        (182, 115, 14): 124,
        (183, 116, 15): 125,
        (183, 118, 15): 126,
        (184, 119, 15): 127,
        (186, 122, 16): 128,
        (187, 123, 16): 130,
        (188, 125, 17): 131,
        (189, 126, 17): 132,
        (190, 127, 18): 133,
        (192, 129, 18): 135,
        (193, 130, 19): 136,
        (194, 131, 20): 137,
        (195, 133, 20): 138,
        (195, 134, 21): 139,
        (196, 136, 22): 140,
        (197, 137, 22): 141,
        (199, 138, 23): 142,
        (200, 139, 24): 143,
        (200, 140, 24): 144,
        (201, 142, 24): 145,
        (202, 143, 25): 146,
        (203, 145, 25): 147,
        (204, 146, 26): 149,
        (206, 147, 27): 150,
        (207, 149, 27): 151,
        (208, 151, 28): 152,
        (209, 152, 28): 153,
        (210, 153, 29): 154,
        (211, 154, 30): 155,
        (212, 156, 31): 156,
        (213, 157, 32): 157,
        (213, 158, 32): 158,
        (215, 160, 34): 160,
        (216, 161, 35): 161,
        (216, 162, 36): 162,
        (217, 164, 38): 163,
        (218, 166, 39): 164,
        (219, 167, 40): 165,
        (220, 168, 41): 166,
        (222, 170, 41): 167,
        (223, 171, 42): 168,
        (224, 172, 43): 169,
        (225, 173, 44): 170,
        (225, 175, 45): 171,
        (227, 177, 47): 173,
        (228, 179, 48): 174,
        (229, 180, 49): 175,
        (229, 181, 50): 176,
        (230, 182, 51): 177,
        (231, 184, 52): 178,
        (232, 185, 53): 179,
        (233, 186, 55): 180,
        (234, 187, 57): 181,
        (235, 189, 59): 182,
        (235, 190, 61): 183,
        (236, 192, 63): 184,
        (236, 194, 65): 186,
        (237, 195, 68): 187,
        (237, 196, 70): 188,
        (237, 197, 72): 189,
        (238, 199, 75): 190,
        (238, 200, 76): 191,
        (239, 201, 79): 192,
        (239, 202, 81): 193,
        (239, 203, 83): 194,
        (240, 205, 86): 195,
        (240, 206, 87): 196,
        (241, 208, 89): 197,
        (241, 210, 91): 199,
        (242, 211, 94): 200,
        (242, 212, 96): 201,
        (243, 213, 99): 202,
        (243, 214, 101): 203,
        (243, 215, 103): 204,
        (244, 217, 105): 205,
        (244, 217, 109): 206,
        (244, 218, 112): 208,
        (245, 220, 117): 209,
        (245, 222, 121): 211,
        (245, 222, 124): 212,
        (246, 223, 127): 213,
        (246, 224, 130): 214,
        (246, 225, 132): 215,
        (247, 226, 136): 216,
        (247, 226, 139): 217,
        (247, 226, 141): 218,
        (247, 227, 145): 219,
        (248, 228, 148): 220,
        (249, 228, 151): 221,
        (249, 229, 154): 222,
        (250, 231, 157): 224,
        (250, 232, 160): 225,
        (250, 232, 162): 226,
        (250, 233, 166): 227,
        (251, 235, 170): 228,
        (251, 236, 172): 229,
        (251, 236, 175): 230,
        (252, 237, 179): 231,
        (252, 237, 180): 232,
        (252, 238, 183): 233,
        (252, 239, 186): 234,
        (252, 240, 192): 237,
        (252, 241, 194): 238,
        (252, 242, 197): 239,
        (252, 242, 200): 240,
        (252, 243, 202): 241,
        (255, 255, 255): 243,
    }


def build_lut(mapping: dict[tuple[int, int, int], int], default_gray=0) -> np.ndarray:
    lut = np.full((1 << 24,), default_gray, dtype=np.uint8)
    keys = np.array(list(mapping.keys()), dtype=np.uint8)
    vals = np.array(list(mapping.values()), dtype=np.uint8)

    # Use vectorized packing for LUT building
    r = keys[:, 0].astype(np.uint32)
    g = keys[:, 1].astype(np.uint32)
    b = keys[:, 2].astype(np.uint32)
    packed = (r << 16) | (g << 8) | b

    lut[packed] = vals
    return lut


def rgb_to_gray_lut(rgb_stack: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    rgb_stack: (N,H,W,3) uint8 (or float that can be cast to uint8)
    returns: (N,H,W) uint8
    """
    # ensure channel-last
    if rgb_stack.ndim == 4 and rgb_stack.shape[-1] != 3 and rgb_stack.shape[1] == 3:
        rgb_stack = np.moveaxis(rgb_stack, 1, -1)

    if rgb_stack.shape[-1] != 3:
        raise ValueError(f"Expected last dim to be 3 (RGB). Got shape={rgb_stack.shape}")

    orig_shape = rgb_stack.shape[:3]

    # cast to uint8 efficiently
    if rgb_stack.dtype == np.uint8:
        rgb_u8 = rgb_stack
    else:
        rgb_u8 = np.clip(rgb_stack, 0, 255).astype(np.uint8, copy=False)

    # Flatten to (N*H*W, 3)
    flat = rgb_u8.reshape(-1, 3)

    # Use Numba-accelerated packing and lookup
    packed = pack_rgb_numba(flat)
    result = lut_lookup_numba(packed, lut)

    return result.reshape(orig_shape)
