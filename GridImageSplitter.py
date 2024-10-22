import torch
import numpy as np
import cv2

class GridImageSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10}),
                "cols": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "split_image"
    CATEGORY = "image/processing"

    def find_split_positions(self, image, num_splits, is_vertical):
        if is_vertical:
            # 对于列，保持原有的边缘检测方法
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges, axis=0)
            
            window_size = len(edge_density) // (num_splits + 1) // 2
            smoothed_density = np.convolve(edge_density, np.ones(window_size)/window_size, mode='same')
            
            split_positions = []
            for i in range(1, num_splits + 1):
                start = i * len(smoothed_density) // (num_splits + 1) - window_size
                end = i * len(smoothed_density) // (num_splits + 1) + window_size
                split = start + np.argmin(smoothed_density[start:end])
                split_positions.append(split)
        else:
            # 对于行，使用均匀分割
            height = image.shape[0]
            split_positions = [i * height // (num_splits + 1) for i in range(1, num_splits + 1)]
        
        return split_positions

    def split_image(self, image, rows, cols):
        print(f"Input image shape: {image.shape}")
        print(f"Input image dtype: {image.dtype}")
        
        # 确保输入图像是 4D tensor (B, H, W, C)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 将图像从 PyTorch tensor 转换为 NumPy 数组
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        height, width = img_np.shape[:2]
        print(f"Original image size: {width}x{height}")

        # 找到分割位置
        vertical_splits = self.find_split_positions(img_np, cols - 1, True)
        horizontal_splits = self.find_split_positions(img_np, rows - 1, False)

        # 创建预览图
        preview_img = image.clone()
        
        # 在预览图上画线
        green_line = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3)
        for x in vertical_splits:
            preview_img[:, :, x:x+2, :] = green_line
        for y in horizontal_splits:
            preview_img[:, y:y+2, :, :] = green_line
        
        # 分割图片
        split_images = []
        h_splits = [0] + horizontal_splits + [height]
        v_splits = [0] + vertical_splits + [width]
        
        # 计算最大的分割尺寸
        max_height = max([h_splits[i+1] - h_splits[i] for i in range(len(h_splits) - 1)])
        max_width = max([v_splits[i+1] - v_splits[i] for i in range(len(v_splits) - 1)])
        
        for i in range(len(h_splits) - 1):
            for j in range(len(v_splits) - 1):
                top = h_splits[i]
                bottom = h_splits[i+1]
                left = v_splits[j]
                right = v_splits[j+1]
                
                cell = image[:, top:bottom, left:right, :]
                
                # 填充到最大尺寸
                pad_bottom = max_height - (bottom - top)
                pad_right = max_width - (right - left)
                cell = torch.nn.functional.pad(cell, (0, 0, 0, pad_right, 0, pad_bottom), mode='constant')
                
                split_images.append(cell)

        split_images = torch.cat(split_images, dim=0)  # 将所有分割图像合并为一个 tensor

        print(f"Preview tensor shape: {preview_img.shape}")
        print(f"Preview tensor dtype: {preview_img.dtype}")
        print(f"Split images tensor shape: {split_images.shape}")
        print(f"Split images tensor dtype: {split_images.dtype}")

        return (preview_img, split_images)

# 这个字典用于注册节点
NODE_CLASS_MAPPINGS = {
    "GridImageSplitter": GridImageSplitter
}
