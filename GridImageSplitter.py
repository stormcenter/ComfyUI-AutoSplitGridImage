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
        处理外部边缘，加强对黑色边框的检测
        """
        if img_np.size == 0 or img_np is None:
            return img_np

        # 转换为多个色彩空间
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        height, width = img_np.shape[:2]
        
        # 最小裁剪量
        min_trim = 18
        # 检查范围扩大到30像素
        check_width = 30
        
        def is_black_region(region):
            """检查区域是否为黑色区域"""
            if len(region.shape) == 3:
                region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            else:
                region_gray = region
                
            # 计算暗色像素的比例
            dark_ratio = np.mean(region_gray < 40)
            # 如果超过60%的像素是暗色的，认为是黑边
            return dark_ratio > 0.6

        def is_white_region(region):
            """检查区域是否为白色区域"""
            if len(region.shape) == 3:
                region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
                sat_mean = np.mean(region_hsv[:,:,1])
                val_mean = np.mean(region_hsv[:,:,2])
                return sat_mean < 30 and val_mean > 225
            return False

        def find_border(gray_img, is_vertical=True, from_start=True):
            """查找边界"""
            if is_vertical:
                total_size = width
                chunk_size = 5  # 每次检查5个像素
            else:
                total_size = height
                chunk_size = 5

            if from_start:
                range_iter = range(0, total_size-chunk_size, chunk_size)
            else:
                range_iter = range(total_size-chunk_size, 0, -chunk_size)

            for i in range_iter:
                if is_vertical:
                    chunk = img_np[:, i:i+chunk_size] if from_start else img_np[:, i-chunk_size:i]
                else:
                    chunk = img_np[i:i+chunk_size, :] if from_start else img_np[i-chunk_size:i, :]
                
                # 分别检查黑边和白边
                if not (is_black_region(chunk) or is_white_region(chunk)):
                    return i if from_start else i
                    
            return min_trim if from_start else total_size - min_trim

        # 检测左边界
        left = find_border(gray, is_vertical=True, from_start=True)
        
        # 检测右边界
        right = find_border(gray, is_vertical=True, from_start=False)
        
        # 检测上边界
        top = find_border(gray, is_vertical=False, from_start=True)
        
        # 检测下边界
        bottom = find_border(gray, is_vertical=False, from_start=False)

        # 强制应用最小裁剪
        # 检查左侧边缘
        left_region = gray[:, :check_width]
        if np.mean(left_region < 40) > 0.3:  # 如果有超过30%的暗色像素
            left = max(left, min_trim)

        # 检查右侧边缘
        right_region = gray[:, -check_width:]
        if np.mean(right_region < 40) > 0.3:  # 如果有超过30%的暗色像素
            right = min(right, width - min_trim)

        # 检查上边缘
        top_region = gray[:check_width, :]
        if np.mean(top_region < 40) > 0.3:
            top = max(top, min_trim)

        # 检查下边缘
        bottom_region = gray[-check_width:, :]
        if np.mean(bottom_region < 40) > 0.3:
            bottom = min(bottom, height - min_trim)

        # 确保裁剪合理
        if (right - left) < width * 0.5 or (bottom - top) < height * 0.5:
            return img_np

        # 应用裁剪
        cropped = img_np[top:bottom, left:right]
        
        # 进行二次检查，确保没有遗漏的黑边
        if cropped.shape[1] > 2 * min_trim:
            right_edge = cv2.cvtColor(cropped[:, -min_trim:], cv2.COLOR_RGB2GRAY)
            if np.mean(right_edge < 40) > 0.3:
                cropped = cropped[:, :-min_trim]
                
        return cropped

    def detect_split_borders(self, img_strip, is_vertical=True):
        """
        检测分割线区域的边框
        """
        hsv = cv2.cvtColor(img_strip, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_strip, cv2.COLOR_RGB2LAB)
        
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        l_channel = lab[:, :, 0]
        
        # 检测白色和黑色区域
        is_border = ((sat < 10) & (val > 248)) | (l_channel < 30)
        
        if is_vertical:
            border_ratios = np.mean(is_border, axis=1)
            indices = np.where(border_ratios < 0.95)[0]
        else:
            border_ratios = np.mean(is_border, axis=0)
            indices = np.where(border_ratios < 0.95)[0]
            
        if len(indices) == 0:
            return 0, img_strip.shape[1] if is_vertical else img_strip.shape[0]
            
        return indices[0], indices[-1]

    def adjust_split_line(self, img_np, split_pos, is_vertical=True, margin=15):
        """
        调整分割线附近的边界
        """
        height, width = img_np.shape[:2]
        
        if is_vertical:
            left_bound = max(0, split_pos - margin)
            right_bound = min(width, split_pos + margin)
            strip = img_np[:, left_bound:right_bound]
            start, end = self.detect_split_borders(strip, False)
            
            start = max(0, start - 5)
            end = min(strip.shape[1], end + 5)
            
            return left_bound + start, left_bound + end
        else:
            top_bound = max(0, split_pos - margin)
            bottom_bound = min(height, split_pos + margin)
            strip = img_np[top_bound:bottom_bound, :]
            start, end = self.detect_split_borders(strip, True)
            
            start = max(0, start - 5)
            end = min(strip.shape[0], end + 5)
            
            return top_bound + start, top_bound + end

    def find_split_positions(self, image, num_splits, is_vertical, split_method):
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
        split_images = []
        max_h = 0
        max_w = 0
        
        # 第一次遍历: 找出所有裁剪后图片的最大宽度和高度
        temp_splits = []
        for i in range(0, len(h_splits)-1, 2):
            for j in range(0, len(v_splits)-1, 2):
                top = h_splits[i]
                bottom = h_splits[i+1]
                left = v_splits[j]
                right = v_splits[j+1]
                
                cell_np = img_np[top:bottom, left:right]
                trimmed_cell = self.remove_external_borders(cell_np)
                temp_splits.append(trimmed_cell)
                
                h, w = trimmed_cell.shape[:2]
                max_h = max(max_h, h)
                max_w = max(max_w, w)
        
        # 第二次遍历: 将所有图片调整为相同尺寸，保持原始比例
        for cell_np in temp_splits:
            h, w = cell_np.shape[:2]
            # 计算缩放比例
            scale = min(max_h/h, max_w/w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # 居中放置
            resized = cv2.resize(cell_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            y_offset = (max_h - new_h) // 2
            x_offset = (max_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # 转换为tensor
            cell_tensor = torch.from_numpy(canvas).float() / 255.0
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
