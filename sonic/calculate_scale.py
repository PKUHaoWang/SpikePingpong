#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算球拍的实际尺寸比例
"""

import cv2
import numpy as np
import os
from get_delta_pos import preprocess_image

def calculate_racket_scale(image_path, output_dir="output", output_prefix="", 
                         color1=(65, 31, 31), tolerance1=25, min_area=100, 
                         center_ratio=0.4, real_width_mm=150, real_height_mm=150):
    """
    计算球拍的实际尺寸比例
    
    参数:
        image_path (str): 输入图像的路径
        output_dir (str): 输出目录
        output_prefix (str): 输出文件名前缀
        color1 (tuple): 球拍颜色的RGB值
        tolerance1 (int): 颜色匹配的容差值
        min_area (int): 最小连通区域面积
        center_ratio (float): 中心区域的比例（0-1之间）
        real_width_mm (float): 球拍实际宽度（毫米）
        real_height_mm (float): 球拍实际高度（毫米）
    
    返回:
        tuple: (1像素对应的毫米数, 球拍中心坐标, 球拍像素尺寸)
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        return None, None, None
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")
    
    # 如果未指定前缀，使用原文件名（不包含路径和扩展名）
    if not output_prefix:
        output_prefix = os.path.splitext(os.path.basename(image_path))[0]
    
    # 裁剪中心区域
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    crop_size = int(min(width, height) * center_ratio)
    x1 = center_x - crop_size // 2
    y1 = center_y - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # 确保裁剪区域在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    # 裁剪图像
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_cropped.jpg"), cropped_image)
    
    # 去噪处理
    denoised_image = preprocess_image(image_path)
    denoised_image = denoised_image[y1:y2, x1:x2]  # 裁剪去噪后的图像
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_denoised.jpg"), denoised_image)
    
    # 转换颜色顺序从RGB到BGR
    color1_bgr = (color1[2], color1[1], color1[0])
    
    print(f"目标颜色 (BGR): {color1_bgr}")
    
    # 创建球拍掩码
    mask1 = cv2.inRange(denoised_image, 
                       np.array([max(0, c - tolerance1) for c in color1_bgr]), 
                       np.array([min(255, c + tolerance1) for c in color1_bgr]))
    
    # 创建中心区域的掩码
    center_mask = np.zeros_like(mask1)
    center_x, center_y = mask1.shape[1] // 2, mask1.shape[0] // 2
    center_size = int(min(mask1.shape[0], mask1.shape[1]) * 0.5)  # 使用50%的中心区域
    x1 = center_x - center_size // 2
    y1 = center_y - center_size // 2
    x2 = x1 + center_size
    y2 = y1 + center_size
    center_mask[y1:y2, x1:x2] = 255
    
    # 只保留中心区域的mask1
    mask1 = cv2.bitwise_and(mask1, center_mask)
    
    # 形态学操作优化mask1
    kernel = np.ones((5,5), np.uint8)  # 使用更大的核来去除小噪点
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)  # 开运算去除小噪点
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)  # 闭运算填充小孔
    
    # 连通区域分析，只保留最大的连通区域
    num_labels1, labels1 = cv2.connectedComponents(mask1)
    filtered_mask1 = np.zeros_like(mask1)
    racket_center = None
    
    if num_labels1 > 1:
        max_area = 0
        max_label = 1
        for label in range(1, num_labels1):
            area = np.sum(labels1 == label)
            if area > max_area:
                max_area = area
                max_label = label
        filtered_mask1[labels1 == max_label] = 255
        
        # 计算球拍中心点
        y_indices, x_indices = np.where(labels1 == max_label)
        racket_center = (int(np.mean(x_indices)), int(np.mean(y_indices)))
        
        # 找到球拍轮廓
        contours, _ = cv2.findContours(filtered_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 获取最小外接矩形
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算矩形的宽度和高度
            width_pixels = rect[1][0]
            height_pixels = rect[1][1]
            
            # 计算1像素对应的毫米数
            mm_per_pixel_width = real_width_mm / width_pixels
            mm_per_pixel_height = real_height_mm / height_pixels
            mm_per_pixel_avg = (mm_per_pixel_width + mm_per_pixel_height) / 2
            
            # 打印结果
            print(f"球拍中心坐标: {racket_center}")
            print(f"球拍像素尺寸: 宽度={width_pixels:.2f}像素, 高度={height_pixels:.2f}像素")
            print(f"1像素对应的毫米数:")
            print(f"  宽度方向: {mm_per_pixel_width:.4f} 毫米/像素")
            print(f"  高度方向: {mm_per_pixel_height:.4f} 毫米/像素")
            print(f"  平均值: {mm_per_pixel_avg:.4f} 毫米/像素")
            
            # 保存掩码
            cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_mask1.jpg"), mask1)
            cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_filtered_mask1.jpg"), filtered_mask1)
            
            # 保存可视化结果
            visualization = denoised_image.copy()
            # 绘制最小外接矩形
            cv2.drawContours(visualization, [box], 0, (0, 255, 0), 2)
            # 绘制中心点
            cv2.circle(visualization, racket_center, 3, (0, 0, 255), -1)  # 减小中心点大小
            cv2.circle(visualization, racket_center, 5, (0, 0, 255), 1)  # 减小外圈大小
            cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_visualization.jpg"), visualization)
            
            return mm_per_pixel_avg, racket_center, (width_pixels, height_pixels)
    
    print("未检测到球拍")
    return None, None, None

def main():
    image_dir = "patimg"
    output_dir = "output_pat"
    color1 = (65, 31, 31)  # RGB
    
    # 获取目录下所有PNG文件
    png_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    if not png_files:
        print(f"在目录 {image_dir} 中没有找到PNG文件")
        return
    
    # 存储所有结果
    all_mm_per_pixel = []
    all_centers = []
    all_sizes = []
    
    # 处理每个PNG文件
    for png_file in png_files:
        image_path = os.path.join(image_dir, png_file)
        print(f"\n处理文件: {png_file}")
        
        mm_per_pixel, center, size = calculate_racket_scale(
            image_path, 
            output_dir, 
            color1=color1,
            tolerance1=15,        # 增大颜色1的容差
            min_area=100, 
            center_ratio=0.4,
            real_width_mm=150,
            real_height_mm=150
        )
        
        if mm_per_pixel is not None:
            all_mm_per_pixel.append(mm_per_pixel)
            all_centers.append(center)
            all_sizes.append(size)
    
    # 计算并打印平均值
    if all_mm_per_pixel:
        avg_mm_per_pixel = sum(all_mm_per_pixel) / len(all_mm_per_pixel)
        avg_center_x = sum(c[0] for c in all_centers) / len(all_centers)
        avg_center_y = sum(c[1] for c in all_centers) / len(all_centers)
        avg_width = sum(s[0] for s in all_sizes) / len(all_sizes)
        avg_height = sum(s[1] for s in all_sizes) / len(all_sizes)
        
        print(f"\n处理完成。共处理 {len(all_mm_per_pixel)} 个文件")
        print(f"平均1像素对应的毫米数: {avg_mm_per_pixel:.4f}")
        print(f"平均球拍中心坐标: ({avg_center_x:.1f}, {avg_center_y:.1f})")
        print(f"平均球拍像素尺寸: ({avg_width:.1f}, {avg_height:.1f})")
    else:
        print("\n没有成功处理任何文件")

if __name__ == "__main__":
    main() 