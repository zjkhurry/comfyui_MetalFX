import MetalFX
import Metal
from Quartz import *
from CoreFoundation import CFArrayGetCount, CFArrayGetValueAtIndex, kCFAllocatorDefault
import numpy as np
from PIL import Image
import ctypes
from typing import Any, Optional


class comfy_MetalFX:
    def __init__(self):
        self.colorPixelFormat = Metal.MTLPixelFormatRGBA8Unorm
        self.desc = None
        self.spatialScaler = None
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.texture = None
        self.output_texture = None
        self.inputWidth = 0
        self.inputHeight = 0
        self.outputWidth = 0
        self.outputHeight = 0
        self.command_queue = self.device.newCommandQueue()
        self.command_buffer = None
        supported = Metal.MTLFXSpatialScalerDescriptor.supportsDevice_(self.device)
        if not supported:
            print("device not supported")
            exit(1)

    def set_Scale(self, inputWidth, inputHeight, outputWidth, outputHeight):
        if inputWidth > inputHeight:
            scale = float(outputWidth) / float(inputWidth)
            outputHeight = int(inputHeight * scale)
        else:
            scale = float(outputHeight) / float(inputHeight)
            outputWidth = int(inputWidth * scale)
        self.desc = MetalFX.MTLFXSpatialScalerDescriptor.alloc().init()
        self.desc.setInputWidth_(inputWidth)
        self.desc.setInputHeight_(inputHeight)
        self.desc.setOutputWidth_(outputWidth)
        self.desc.setOutputHeight_(outputHeight)
        self.desc.setColorTextureFormat_(self.colorPixelFormat)
        self.desc.setOutputTextureFormat_(self.colorPixelFormat)
        self.desc.setColorProcessingMode_(
            MetalFX.MTLFXSpatialScalerColorProcessingMode(0)
        )
        self.spatialScaler = self.desc.newSpatialScalerWithDevice_(self.device)
        texture_descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            self.colorPixelFormat, inputWidth, inputHeight, False
        )
        self.texture = self.device.newTextureWithDescriptor_(texture_descriptor)
        output_descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            self.colorPixelFormat, outputWidth, outputHeight, False
        )
        self.output_texture = self.device.newTextureWithDescriptor_(output_descriptor)
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight

    def set_inputImage(self, img, outputWidth, outputHeight):
        if img.ndim != 3 or img.shape[2] != 4:
            texture_data = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            texture_data[:, :, :3] = img
            texture_data[:, :, 3] = 255
        else:
            texture_data = img
        width = texture_data.shape[1]
        height = texture_data.shape[0]
        if (
            width != self.inputWidth
            or height != self.inputHeight
            or outputWidth != self.outputWidth
            or outputHeight != self.outputHeight
        ):
            self.set_Scale(width, height, outputWidth, outputHeight)
        region = Metal.MTLRegion(
            (0, 0, 0), (self.texture.width(), self.texture.height(), 1)
        )
        bytes_per_row = self.texture.width() * 4

        self.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region, 0, texture_data, bytes_per_row  # mipmapLevel
        )
        return self.rander()

    def rander(self):
        self.spatialScaler.setColorTexture_(self.texture)
        self.spatialScaler.setInputContentHeight_(self.inputHeight)
        self.spatialScaler.setInputContentWidth_(self.inputWidth)
        self.spatialScaler.setOutputTexture_(self.output_texture)

        self.command_buffer = self.command_queue.commandBuffer()
        self.spatialScaler.encodeToCommandBuffer_(self.command_buffer)
        self.command_buffer.commit()
        self.command_buffer.waitUntilCompleted()
        if (
            self.output_texture.pixelFormat() != Metal.MTLPixelFormatBGRA8Unorm
            and self.output_texture.pixelFormat() != Metal.MTLPixelFormatRGBA8Unorm
        ):
            raise Exception(
                "Not correct pixel format (expected MTLPixelFormatBGRA8Unorm or MTLPixelFormatRGBA8Unorm)"
            )

        bytes_per_row = self.output_texture.width() * 4
        bytes_per_image = bytes_per_row * self.output_texture.height()
        mipmap_level = 0
        slice_number = 0
        region = Metal.MTLRegionMake2D(
            0, 0, self.output_texture.width(), self.output_texture.height()
        )

        # if buffer is None:
        buffer = ctypes.create_string_buffer(bytes_per_image)

        if len(buffer) != bytes_per_image:
            raise Exception(
                f"Buffer is not big enough (expected: {bytes_per_image}, actual: {len(buffer)})"
            )

        self.output_texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice_(
            buffer, bytes_per_row, bytes_per_image, region, mipmap_level, slice_number
        )

        raw_bytes = bytes(buffer.raw)
        return Image.frombytes(
            "RGBA",
            (self.output_texture.width(), self.output_texture.height()),
            raw_bytes,
        )
