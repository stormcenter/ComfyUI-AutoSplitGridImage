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

    def trim_white_borders(self, img_np):
        """
        增强的白边裁剪算法，特别优化了对天空等浅色区域的处理
        """
        if len(img_np.shape) == 3:
            # 转换为多个颜色空间以更好地检测边界
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            
            # 提取各个通道
            v_channel = hsv[:,:,2]
            l_channel = lab[:,:,0]
            
            # 计算梯度
            gradient_v = cv2.Sobel(v_channel, cv2.CV_64F, 1, 1, ksize=3)
            gradient_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1, ksize=3)
            
            # 综合多个指标
            gradient_magnitude = np.sqrt(gradient_v**2 + gradient_l**2)
            
            # 创建掩码
            mask = np.zeros_like(v_channel)
            
            # 基于梯度和亮度的自适应阈值
            gradient_threshold = np.percentile(gradient_magnitude, 95) * 0.1
            brightness_threshold = np.percentile(v_channel, 95) * 0.95
            
            # 更新掩码
            mask[gradient_magnitude > gradient_threshold] = 255
            mask[v_channel < brightness_threshold] = 255
            
            # 应用形态学操作
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)
        else:
            # 如果是灰度图像
            gradient = cv2.Sobel(img_np, cv2.CV_64F, 1, 1, ksize=3)
            mask = (gradient > np.percentile(gradient, 95) * 0.1).astype(np.uint8) * 255

        # 找到非边界区域
        coords = cv2.findNonZero(mask)
        if coords is None:
            return img_np
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # 智能边界调整
        def adjust_boundary(start, end, size, is_start=True):
            if is_start:
                while start < end and start < size - 1:
                    if np.any(mask[start:start+1, :] > 0):
                        break
                    start += 1
                return max(0, start - 1)
            else:
                while end > start and end > 0:
                    if np.any(mask[end-1:end, :] > 0):
                        break
                    end -= 1
                return min(size, end + 1)
        
        # 调整边界
        y = adjust_boundary(y, y+h, img_np.shape[0], True)
        h = adjust_boundary(y, y+h, img_np.shape[0], False) - y
        x = adjust_boundary(x, x+w, img_np.shape[1], True)
        w = adjust_boundary(x, x+w, img_np.shape[1], False) - x

        # 裁剪图像
        cropped = img_np[y:y+h, x:x+w]
        
        # 边缘检查和优化
        if cropped.shape[0] > 10 and cropped.shape[1] > 10:
            edge_width = 3
            edges = {
                'top': cropped[:edge_width, :],
                'bottom': cropped[-edge_width:, :],
                'left': cropped[:, :edge_width],
                'right': cropped[:, -edge_width:]
            }
            
            for edge_name, edge in edges.items():
                if len(edge.shape) == 3:
                    edge_gradient = cv2.Sobel(cv2.cvtColor(edge, cv2.COLOR_RGB2LAB)[:,:,0], 
                                            cv2.CV_64F, 1, 1, ksize=3)
                else:
                    edge_gradient = cv2.Sobel(edge, cv2.CV_64F, 1, 1, ksize=3)
                
                if np.mean(np.abs(edge_gradient)) < gradient_threshold:
                    if edge_name == 'top' and y + edge_width < y + h:
                        cropped = cropped[edge_width:, :]
                    elif edge_name == 'bottom' and h - edge_width > 0:
                        cropped = cropped[:-edge_width, :]
                    elif edge_name == 'left' and x + edge_width < x + w:
                        cropped = cropped[:, edge_width:]
                    elif edge_name == 'right' and w - edge_width > 0:
                        cropped = cropped[:, :-edge_width]

        return cropped

    def find_split_positions(self, image, num_splits, is_vertical, split_method):
        if split_method == "uniform":
            size = image.shape[1] if is_vertical else image.shape[0]
            return [i * size // (num_splits + 1) for i in range(1, num_splits + 1)]
        else:  # edge_detection
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
        print(f"Input image shape: {image.shape}")
        print(f"Input image dtype: {image.dtype}")
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        height, width = img_np.shape[:2]
        print(f"Original image size: {width}x{height}")

        vertical_splits = self.find_split_positions(img_np, cols - 1, True, col_split_method)
        horizontal_splits = self.find_split_positions(img_np, rows - 1, False, row_split_method)

        preview_img = image.clone()
        
        green_line = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3)
        for x in vertical_splits:
            preview_img[:, :, x:x+2, :] = green_line
        for y in horizontal_splits:
            preview_img[:, y:y+2, :, :] = green_line
        
        split_images = []
        h_splits = [0] + horizontal_splits + [height]
        v_splits = [0] + vertical_splits + [width]
        
        max_height = 0
        max_width = 0
        temp_splits = []
        
        for i in range(len(h_splits) - 1):
            for j in range(len(v_splits) - 1):
                top = h_splits[i]
                bottom = h_splits[i+1]
                left = v_splits[j]
                right = v_splits[j+1]
                
                cell_np = img_np[top:bottom, left:right]
                trimmed_cell = self.trim_white_borders(cell_np)
                
                max_height = max(max_height, trimmed_cell.shape[0])
                max_width = max(max_width, trimmed_cell.shape[1])
                
                temp_splits.append(trimmed_cell)
        
        for trimmed_cell in temp_splits:
            cell_tensor = torch.from_numpy(trimmed_cell).float() / 255.0
            cell_tensor = cell_tensor.unsqueeze(0)
            
            pad_bottom = max_height - trimmed_cell.shape[0]
            pad_right = max_width - trimmed_cell.shape[1]
            cell_tensor = torch.nn.functional.pad(cell_tensor, (0, 0, 0, pad_right, 0, pad_bottom), mode='constant')
            
            split_images.append(cell_tensor)

        split_images = torch.cat(split_images, dim=0)

        print(f"Preview tensor shape: {preview_img.shape}")
        print(f"Preview tensor dtype: {preview_img.dtype}")
        print(f"Split images tensor shape: {split_images.shape}")
        print(f"Split images tensor dtype: {split_images.dtype}")

        return (preview_img, split_images)

NODE_CLASS_MAPPINGS = {
    "GridImageSplitter": GridImageSplitter
}
