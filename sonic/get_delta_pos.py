#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像颜色分割：从图片中分割出红色和黄色区域
"""

import cv2
import numpy as np
import os
import argparse
from skimage.restoration import denoise_tv_chambolle


def preprocess_image(image_path, method='median'):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    if method == 'fastnlmeans':
        # 非局部均值去噪
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif method == 'gaussian':
        # 高斯滤波去噪
        denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        # 中值滤波去噪
        denoised_image = cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        # 双边滤波去噪
        denoised_image = cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'tv':
        # TV去噪
        denoised_image = denoise_tv_chambolle(image, weight=0.1, multichannel=True)
        denoised_image = (denoised_image * 255).astype(np.uint8)
    else:
        raise ValueError(f"不支持的去噪方法: {method}")
    
    return denoised_image


def segment_colors(image_path, output_dir="output", output_prefix="", color1=(141,84,57), color2=(76,57,34), tolerance1=15, tolerance2=15, min_area=100, center_ratio=0.6, mm_per_pixel=2.3108):
    """
    从图片中分割出球拍区域，并将球拍内部的区域作为球，计算它们的中心位置
    
    参数:
        image_path (str): 输入图像的路径
        output_dir (str): 输出目录
        output_prefix (str): 输出文件名前缀
        color1 (tuple): 第一个目标颜色的RGB值
        color2 (tuple): 第二个目标颜色的RGB值
        tolerance1 (int): 第一个颜色匹配的容差值
        tolerance2 (int): 第二个颜色匹配的容差值
        min_area (int): 最小连通区域面积
        center_ratio (float): 中心区域的比例（0-1之间）
        mm_per_pixel (float): 1像素对应的毫米数
    
    返回:
        tuple: (球拍中心坐标, 球的相对坐标列表(毫米))
        坐标系统：
        - x轴：向右为正
        - y轴：向上为正
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        return None, None
    
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
    color2_bgr = (color2[2], color2[1], color2[0])
    
    print(f"目标颜色1 (BGR): {color1_bgr}")
    print(f"目标颜色2 (BGR): {color2_bgr}")
    
    # 创建第一个颜色的掩码
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
    ball_centers = []
    ball_relative_positions = []
    
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
            # 创建球拍内部区域的掩码
            ball_mask = np.zeros_like(filtered_mask1)
            cv2.drawContours(ball_mask, contours, -1, 255, -1)  # 填充轮廓内部
            
            # 从内部区域中排除球拍本身
            ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(filtered_mask1))
            
            # 形态学操作优化球的掩码
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
            
            # 连通区域分析，找到所有可能的球
            num_labels2, labels2 = cv2.connectedComponents(ball_mask)
            filtered_mask2 = np.zeros_like(ball_mask)
            
            for label in range(1, num_labels2):
                area = np.sum(labels2 == label)
                if area >= min_area:
                    filtered_mask2[labels2 == label] = 255
                    # 计算这个区域的中心点
                    y_indices, x_indices = np.where(labels2 == label)
                    center_y = int(np.mean(y_indices))
                    center_x = int(np.mean(x_indices))
                    ball_centers.append((center_x, center_y))
            
            # 如果找到了球拍中心的空洞，就不需要进行mask2的检测
            if ball_centers:
                # 计算球的相对坐标（以球拍中心为原点，单位为毫米）
                for center in ball_centers:
                    # 计算相对像素坐标
                    rel_x = center[0] - racket_center[0]
                    rel_y = racket_center[1] - center[1]  # 注意这里取负，使y轴向上为正
                    # 转换为毫米
                    rel_x_mm = rel_x * mm_per_pixel
                    rel_y_mm = rel_y * mm_per_pixel
                    ball_relative_positions.append((rel_x_mm, rel_y_mm))
                
                # 保存掩码
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_mask1.jpg"), mask1)
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_filtered_mask1.jpg"), filtered_mask1)
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_ball_mask.jpg"), ball_mask)
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_filtered_mask2.jpg"), filtered_mask2)
                
                # 创建分割后的图像
                segmented1 = cv2.bitwise_and(denoised_image, denoised_image, mask=filtered_mask1)
                segmented2 = cv2.bitwise_and(denoised_image, denoised_image, mask=filtered_mask2)
                
                # 创建组合图像，使用不同颜色显示两个区域
                segmented_combined = denoised_image.copy()
                # 将mask1区域显示为红色
                segmented_combined[filtered_mask1 > 0] = [0, 0, 255]  # BGR格式，红色
                # 将mask2区域显示为绿色
                segmented_combined[filtered_mask2 > 0] = [0, 255, 0]  # BGR格式，绿色
                
                # 保存分割后的图像
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_segmented1.jpg"), segmented1)
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_segmented2.jpg"), segmented2)
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_segmented_combined.jpg"), segmented_combined)
                
                # 保存一个可视化结果，显示原始图像和中心点
                visualization = denoised_image.copy()
                # 在球拍中心画小红点
                cv2.circle(visualization, racket_center, 3, (0, 0, 255), -1)  # 减小中心点大小
                cv2.circle(visualization, racket_center, 5, (0, 0, 255), 1)  # 减小外圈大小
                
                # 在球中心画小绿点
                for center in ball_centers:
                    cv2.circle(visualization, center, 2, (0, 255, 0), -1)
                
                # 只添加相对偏移量标注
                for i, pos in enumerate(ball_relative_positions):
                    cv2.putText(visualization, f"({pos[0]:.1f},{pos[1]:.1f})", 
                               (ball_centers[i][0]+5, ball_centers[i][1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_visualization.jpg"), visualization)
                
                # 打印中心点坐标
                print(f"球拍中心坐标: {racket_center}")
                print(f"球中心坐标列表: {ball_centers}")
                print("\n以球拍中心为原点的球的相对坐标（毫米）:")
                print("坐标系统：x轴向右为正，y轴向上为正")
                for i, pos in enumerate(ball_relative_positions):
                    print(f"球 {i+1}: x={pos[0]:.2f}mm, y={pos[1]:.2f}mm")
                
                return racket_center, ball_relative_positions
    
    # 如果没有找到球拍中心的空洞，则继续使用原来的mask2检测方法
    # 创建第二个颜色的掩码
    mask2 = cv2.inRange(denoised_image, 
                       np.array([max(0, c - tolerance2) for c in color2_bgr]), 
                       np.array([min(255, c + tolerance2) for c in color2_bgr]))
    
    # 只保留中心区域的mask2
    mask2 = cv2.bitwise_and(mask2, center_mask)
    
    # 获取球拍区域（mask1的最大连通区域）
    racket_region = np.zeros_like(mask1)
    num_labels1, labels1 = cv2.connectedComponents(mask1)
    if num_labels1 > 1:
        max_area = 0
        max_label = 1
        for label in range(1, num_labels1):
            area = np.sum(labels1 == label)
            if area > max_area:
                max_area = area
                max_label = label
        racket_region[labels1 == max_label] = 255
    
    # 扩展球拍区域，确保包含附近的球（20像素范围）
    racket_region = cv2.dilate(racket_region, np.ones((41,41), np.uint8))  # 使用41x41的核，确保20像素的扩展
    
    # 在球拍附近寻找球
    filtered_mask2 = np.zeros_like(mask2)
    num_labels2, labels2 = cv2.connectedComponents(mask2)
    
    # 找到球拍的中心点
    racket_center = None
    if num_labels1 > 1:
        max_area = 0
        max_label = 1
        for label in range(1, num_labels1):
            area = np.sum(labels1 == label)
            if area > max_area:
                max_area = area
                max_label = label
        # 计算球拍中心点
        y_indices, x_indices = np.where(labels1 == max_label)
        racket_center = (int(np.mean(x_indices)), int(np.mean(y_indices)))
    
    if racket_center is not None:
        min_distance = float('inf')
        closest_label = None
        
        for label in range(1, num_labels2):
            # 获取当前连通区域
            current_mask = np.zeros_like(mask2)
            current_mask[labels2 == label] = 255
            
            # 检查是否在球拍20像素范围内
            if np.any(cv2.bitwise_and(current_mask, racket_region)):
                # 计算圆形度
                contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # 只保留接近圆形的区域
                        if circularity > 0.6:  # 圆形度阈值
                            # 计算当前区域中心点
                            y_indices, x_indices = np.where(labels2 == label)
                            center_y = int(np.mean(y_indices))
                            center_x = int(np.mean(x_indices))
                            
                            # 计算到球拍中心的距离
                            distance = np.sqrt((center_x - racket_center[0])**2 + (center_y - racket_center[1])**2)
                            
                            # 更新最近区域
                            if distance < min_distance:
                                min_distance = distance
                                closest_label = label
        
        # 只保留最近的那块区域
        if closest_label is not None:
            filtered_mask2[labels2 == closest_label] = 255
    
    # 合并两个掩码
    combined_mask = cv2.bitwise_or(filtered_mask1, filtered_mask2)
    
    # 创建球的掩码（球拍内部的区域）
    # 直接使用filtered_mask2来找到球的位置
    num_labels2, labels2 = cv2.connectedComponents(filtered_mask2)
    ball_centers = []
    
    if num_labels2 > 1:
        for label in range(1, num_labels2):
            area = np.sum(labels2 == label)
            if area >= min_area:
                # 计算这个区域的中心点
                y_indices, x_indices = np.where(labels2 == label)
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                ball_centers.append((center_x, center_y))
    
    # 计算球的相对坐标（以球拍中心为原点，单位为毫米）
    ball_relative_positions = []
    for center in ball_centers:
        # 计算相对像素坐标
        rel_x = center[0] - racket_center[0]
        rel_y = racket_center[1] - center[1]  # 注意这里取负，使y轴向上为正
        # 转换为毫米
        rel_x_mm = rel_x * mm_per_pixel
        rel_y_mm = rel_y * mm_per_pixel
        ball_relative_positions.append((rel_x_mm, rel_y_mm))
    
    # 打印中心点坐标
    print(f"球拍中心坐标: {racket_center}")
    print(f"球中心坐标列表: {ball_centers}")
    print("\n以球拍中心为原点的球的相对坐标（毫米）:")
    print("坐标系统：x轴向右为正，y轴向上为正")
    for i, pos in enumerate(ball_relative_positions):
        print(f"球 {i+1}: x={pos[0]:.2f}mm, y={pos[1]:.2f}mm")
    
    # 保存掩码
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_mask1.jpg"), mask1)
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_mask2.jpg"), mask2)
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_filtered_mask2.jpg"), filtered_mask2)  # 保存处理后的mask2
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_combined_mask.jpg"), combined_mask)
    
    # 创建分割后的图像
    segmented1 = cv2.bitwise_and(denoised_image, denoised_image, mask=filtered_mask1)
    segmented2 = cv2.bitwise_and(denoised_image, denoised_image, mask=filtered_mask2)
    
    # 创建组合图像，使用不同颜色显示两个区域
    segmented_combined = denoised_image.copy()
    # 将mask1区域显示为红色
    segmented_combined[filtered_mask1 > 0] = [0, 0, 255]  # BGR格式，红色
    # 将mask2区域显示为绿色
    segmented_combined[filtered_mask2 > 0] = [0, 255, 0]  # BGR格式，绿色
    
    # 保存分割后的图像
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_segmented1.jpg"), segmented1)
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_segmented2.jpg"), segmented2)
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_segmented_combined.jpg"), segmented_combined)
    
    # 保存一个可视化结果，显示原始图像和中心点
    visualization = denoised_image.copy()
    # 在球拍中心画小红点
    cv2.circle(visualization, racket_center, 3, (0, 0, 255), -1)  # 减小中心点大小
    cv2.circle(visualization, racket_center, 5, (0, 0, 255), 1)  # 减小外圈大小
    
    # 在球中心画小绿点
    for center in ball_centers:
        cv2.circle(visualization, center, 2, (0, 255, 0), -1)
    
    # 只添加相对偏移量标注
    for i, pos in enumerate(ball_relative_positions):
        cv2.putText(visualization, f"({pos[0]:.1f},{pos[1]:.1f})", 
                   (ball_centers[i][0]+5, ball_centers[i][1]+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_visualization.jpg"), visualization)
    
    return racket_center, ball_relative_positions

def main():
    image_path = "797.png"
    output_dir = "output"
    color1 = (65, 31, 31)  # RGB
    color2 = (79, 58, 34)  # RGB，更新目标颜色2
    mm_per_pixel = 2.3108  # 1像素对应的毫米数
    racket_center, ball_positions = segment_colors(
        image_path, 
        output_dir, 
        color1=color1,
        color2=color2,
        tolerance1=25,   # 增大颜色1的容差
        tolerance2=12,   # 减小颜色2的容差，使检测更精确
        min_area=100,   
        center_ratio=0.4,
        mm_per_pixel=mm_per_pixel
    )
    print(f"\n处理完成。输出目录：{output_dir}")
    print(f"球拍中心坐标: {racket_center}")
    print("\n球的相对坐标（毫米）:")
    print("坐标系统：x轴向右为正，y轴向上为正")
    for i, pos in enumerate(ball_positions):
        print(f"球 {i+1}: x={pos[0]:.2f}mm, y={pos[1]:.2f}mm")

if __name__ == "__main__":
    main()