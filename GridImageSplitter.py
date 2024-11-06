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
                "row_split_method": (["uniform", "edge_detection"],),
                "col_split_method": (["uniform", "edge_detection"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "split_image"
    CATEGORY = "image/processing"

    def remove_external_borders(self, img_np):
        """
        处理图片外部边缘的白边
        """
        if img_np.size == 0 or img_np is None:
            return img_np

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # 获取饱和度和亮度通道
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # 定义白色区域的条件
        is_white = (sat < 25) & (val > 230)
        
        height, width = img_np.shape[:2]
        
        # 查找非白色区域的边界
        y_nonwhite = np.where(np.any(~is_white, axis=1))[0]
        x_nonwhite = np.where(np.any(~is_white, axis=0))[0]
        
        if len(y_nonwhite) == 0 or len(x_nonwhite) == 0:
            return img_np
            
        # 获取边界
        top = y_nonwhite[0]
        bottom = y_nonwhite[-1] + 1
        left = x_nonwhite[0]
        right = x_nonwhite[-1] + 1
        
        # 添加小边距
        margin = 1
        top = max(0, top - margin)
        bottom = min(height, bottom + margin)
        left = max(0, left - margin)
        right = min(width, right + margin)
        
        # 裁剪图像
        cropped = img_np[top:bottom, left:right]
        
        return cropped

    def detect_white_border(self, img_strip, is_vertical=True):
        """
        检测条带中的白色边界，使用更保守的阈值
        """
        hsv = cv2.cvtColor(img_strip, cv2.COLOR_RGB2HSV)
        
        # 获取饱和度和亮度
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # 使用更严格的白色区域条件
        is_white = (sat < 10) & (val > 248)  # 更严格的条件
        
        if is_vertical:
            white_ratios = np.mean(is_white, axis=1)
            indices = np.where(white_ratios < 0.95)[0]  # 更高的阈值
        else:
            white_ratios = np.mean(is_white, axis=0)
            indices = np.where(white_ratios < 0.95)[0]  # 更高的阈值
            
        if len(indices) == 0:
            return 0, img_strip.shape[1] if is_vertical else img_strip.shape[0]
            
        return indices[0], indices[-1]

    def adjust_split_line(self, img_np, split_pos, is_vertical=True, margin=15):
        """
        调整分割线附近的白色边界，使用更小的检测范围
        """
        height, width = img_np.shape[:2]
        
        if is_vertical:
            left_bound = max(0, split_pos - margin)
            right_bound = min(width, split_pos + margin)
            strip = img_np[:, left_bound:right_bound]
            start, end = self.detect_white_border(strip, False)
            
            # 添加偏移以保留更多内容
            start = max(0, start - 5)  # 向外扩展几个像素
            end = min(strip.shape[1], end + 5)
            
            return left_bound + start, left_bound + end
        else:
            top_bound = max(0, split_pos - margin)
            bottom_bound = min(height, split_pos + margin)
            strip = img_np[top_bound:bottom_bound, :]
            start, end = self.detect_white_border(strip, True)
            
            # 添加偏移以保留更多内容
            start = max(0, start - 5)  # 向外扩展几个像素
            end = min(strip.shape[0], end + 5)
            
            return top_bound + start, top_bound + end

    def find_split_positions(self, image, num_splits, is_vertical, split_method):
        """
        找到分割位置
        """
        if split_method == "uniform":
            size = image.shape[1] if is_vertical else image.shape[0]
            return [i * size // (num_splits + 1) for i in range(1, num_splits + 1)]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges, axis=0) if is_vertical else np.sum(edges, axis=1)
            
            window_size = len(edge_density) // (num_splits + 1) // 2
            smoothed_density = np.convolve(edge_density, np.ones(window_size)/window_size, mode='same')
            
            split_positions = []
            for i in range(1, num_splits + 1):
                start = i * len(smoothed_density) // (num_splits + 1) - window_size
                end = i * len(smoothed_density) // (num_splits + 1) + window_size
                split = start + np.argmin(smoothed_density[start:end])
                split_positions.append(split)
            return split_positions

    def process_second_row(self, cells):
        """
        处理第二行图片 - 固定裁剪上部30像素
        """
        processed_cells = []
        trim_pixels = 30  # 固定裁剪像素数
        
        for cell in cells:
            height = cell.shape[0]
            if height > trim_pixels:
                # 从第30个像素开始裁剪
                cropped = cell[trim_pixels:, :]
                processed_cells.append(cropped)
            else:
                processed_cells.append(cell)
        return processed_cells

    def split_image(self, image, rows, cols, row_split_method, col_split_method):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        height, width = img_np.shape[:2]

        # 获取分割位置
        vertical_splits = self.find_split_positions(img_np, cols - 1, True, col_split_method)
        horizontal_splits = self.find_split_positions(img_np, rows - 1, False, row_split_method)

        # 创建预览图
        preview_img = image.clone()
        green_line = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3)
        for x in vertical_splits:
            preview_img[:, :, x:x+2, :] = green_line
        for y in horizontal_splits:
            preview_img[:, y:y+2, :, :] = green_line

        # 调整分割位置
        adjusted_v_splits = []
        for split in vertical_splits:
            left, right = self.adjust_split_line(img_np, split, True)
            adjusted_v_splits.extend([left, right])
            
        adjusted_h_splits = []
        for split in horizontal_splits:
            top, bottom = self.adjust_split_line(img_np, split, False)
            adjusted_h_splits.extend([top, bottom])

        # 获取分割边界
        h_splits = [0] + sorted(adjusted_h_splits) + [height]
        v_splits = [0] + sorted(adjusted_v_splits) + [width]

        # 处理所有分割区域
        original_cells = []
        second_row_cells = []
        max_ratio = 0
        
        for i in range(0, len(h_splits)-1, 2):
            row_cells = []
            for j in range(0, len(v_splits)-1, 2):
                top = h_splits[i]
                bottom = h_splits[i+1]
                left = v_splits[j]
                right = v_splits[j+1]
                
                cell_np = img_np[top:bottom, left:right]
                trimmed_cell = self.remove_external_borders(cell_np)
                
                # 存储原始裁剪结果
                original_cells.append(trimmed_cell)
                
                # 如果是第二行，存储额外的单元格
                if i == 2:  # 第二行
                    row_cells.append(trimmed_cell)
                
                # 更新最大比例
                h, w = trimmed_cell.shape[:2]
                ratio = w / h
                max_ratio = max(max_ratio, ratio)

        # 处理第二行的额外裁剪
        if row_cells:
            second_row_processed = self.process_second_row(row_cells)
            # 更新最大比例
            for cell in second_row_processed:
                h, w = cell.shape[:2]
                ratio = w / h
                max_ratio = max(max_ratio, ratio)
            second_row_cells.extend(second_row_processed)

        # 合并所有需要处理的图片
        all_cells = original_cells + second_row_cells

        # 调整大小和格式转换
        target_height = 1024
        target_width = int(target_height * max_ratio)
        target_width = (target_width + 1) & ~1

        split_images = []
        for cell_np in all_cells:
            resized = cv2.resize(cell_np, (target_width, target_height), 
                               interpolation=cv2.INTER_LANCZOS4)
            cell_tensor = torch.from_numpy(resized).float() / 255.0
            cell_tensor = cell_tensor.unsqueeze(0)
            split_images.append(cell_tensor)

        stacked_images = torch.cat(split_images, dim=0)
        
        if stacked_images.shape[-1] != 3:
            stacked_images = stacked_images.permute(0, 2, 3, 1)

        print(f"Final stacked shape: {stacked_images.shape}")
        
        return (preview_img, stacked_images)

NODE_CLASS_MAPPINGS = {
    "GridImageSplitter": GridImageSplitter
}
