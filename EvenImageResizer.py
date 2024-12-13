import torch

class EvenImageResizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_to_even"
    CATEGORY = "image/processing"

    def resize_to_even(self, image):
        # 确保图片是 4D tensor [batch, height, width, channels]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        height, width = image.shape[1:3]
        
        # 计算新的高度和宽度
        new_height = height - (height % 2)
        new_width = width - (width % 2)
        
        # 如果尺寸没有变化，直接返回原图
        if new_height == height and new_width == width:
            return (image,)
        
        # 裁剪图片以确保宽高为偶数
        resized_image = image[:, :new_height, :new_width, :]
        
        return (resized_image,)

NODE_CLASS_MAPPINGS = {
    "EvenImageResizer": EvenImageResizer
} 