# utils_test.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import ADNI_MRI  # 假设你的主代码文件名是 dataset.py
import random

def visualize_augmentation():
    # 随机生成一个模拟的 3D 图像（5个切片，每个切片128x128）
    n_slices = 5
    img_size = 128
    fake_img = np.random.randint(0, 256, size=(n_slices, img_size, img_size)).astype(np.uint8)

    print(f"原始图像尺寸: {fake_img.shape}")

    # 创建 ADNI_MRI 实例（这里只为了调用 augmentation_2d）
    dummy_dataset = ADNI_MRI(image_dir='', nserial=n_slices, mode='train')

    # 使用 2D 增广函数处理
    aug_img = dummy_dataset.augmentation_2d(np.copy(fake_img))

    print(f"增广后图像尺寸: {aug_img.shape}")

    # 可视化对比：原始第0层 vs 增广第0层
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(fake_img[0], cmap='gray')
    axs[0].set_title("原始图像 Slice 0")
    axs[1].imshow(aug_img[0, 0].numpy(), cmap='gray')
    axs[1].set_title("增广后图像 Slice 0")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_augmentation()
