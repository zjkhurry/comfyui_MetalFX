from .utils import *
import torch
import numpy as np
from tqdm import tqdm


class metalFX_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_width": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 8192, "step": 1},
                ),
                "output_heght": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 8192, "step": 1},
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
                "mask": ("MASK",),
                "video_info": ("VHS_VIDEOINFO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "mask", "audio", "video_info")
    FUNCTION = "apply_metalFX"
    CATEGORY = "MetalFX"
    TITLE = "MetalFX Upsacle"

    def apply_metalFX(
        self, image, output_width, output_heght, audio=None, mask=None, video_info=None
    ):

        m = comfy_MetalFX()
        output_list = []
        mask_output_list = []
        for a, img in tqdm(
            enumerate(image), total=len(image), desc="Processing frames", leave=False
        ):
            i = 255.0 * img.cpu().numpy()
            output = m.set_inputImage(i, output_width, output_heght)
            output_np = np.array(output).astype(np.float32) / 255.0
            output_list.append(torch.from_numpy(output_np)[None,])

            if mask is not None:
                msk = mask.cpu().numpy()[a] * 255.0
                msk = msk.astype(np.uint8)  # Alpha通道
                mask_rgba = np.zeros((msk.shape[0], msk.shape[1], 4), dtype=np.uint8)
                mask_rgba[:, :, :3] = msk[:, :, None]  # 复制mask到RGB通道
                mask_rgba[:, :, 3] = 255  # 设置Alpha通道
                mask_output = m.set_inputImage(mask_rgba, output_width, output_heght)
                mask_output_np = np.array(mask_output).astype(np.float32) / 255.0
                mask_output_list.append(torch.from_numpy(mask_output_np)[None,])

        output_tensor = torch.cat(output_list, dim=0)
        final_tensor = output_tensor[:, :, :, :3]
        if mask is not None:
            mask_tensor = torch.cat(mask_output_list, dim=0)
            mask = mask_tensor[:, :, :, 0]
        return (final_tensor, mask, audio, video_info)
