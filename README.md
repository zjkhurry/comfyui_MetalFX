# ComfyUI MetalFX

A custom node for ComfyUI that enables high-quality image and video upscaling using Apple MetalFX technology.

## Features

- High-quality image and video upscaling
- Hardware-accelerated using Apple MetalFX 
- Support for both image and mask processing

## Requirements

- macOS >= 13.0
- Apple device with Metal support
- Python 3.x
- ComfyUI

## Installation

```bash
cd custom_nodes
git clone https://github.com/zjkhurry/comfyui_MetalFX.git
cd comfyui_MetalFX
pip install -r requirements.txt
```

## Usage

1. Add "MetalFX Upscale" node in ComfyUI
2. Connect input image/video
3. Set target output resolution 
4. Run the workflow

Upscale image:
![upscale image](./imgs/image.jpeg)
Upscale video:
![upscale video](./imgs/video.png)


### Input Parameters

- `image`: Input image or video frames
- `output_width`: Output width (1-8192)
- `output_height`: Output height (1-8192)
- `mask`: (Optional) Input mask
- `audio`: (Optional) Input audio
- `video_info`: (Optional) Video information

## License

MIT License

## Acknowledgments 

- [Apple MetalFX](https://developer.apple.com/documentation/metalfx?language=objc)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)