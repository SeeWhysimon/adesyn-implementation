import types
import os
import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
import adesyn.utils as utils

import os
import numpy as np
import random

def generate_fake_npy_dataset(
    per_class_num=30,
    img_size=192,
    img_depth=160,
    save_dir="../data/NP_3D/setA",
    classes=["AD", "MCI", "NM"]
):
    print(f"🚧 开始生成数据，每类 {per_class_num} 张，总共 {per_class_num * len(classes)} 张")

    for cls in classes:
        class_dir = os.path.join(save_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(per_class_num):
            fake_data = np.random.randint(
                0, 256, size=(img_depth, img_size, img_size)
            ).astype(np.uint8)

            out_path = os.path.join(class_dir, f"fake_{cls}_{i:03d}.npy")
            np.save(out_path, fake_data)

            print(f"✅ 已保存：{out_path}")

    print(f"\n🎉 所有类别生成完毕，共 {len(classes)} 类 × {per_class_num} 张 = {per_class_num * len(classes)} 张")

def medi_imread_test():
    image_dir = "../data"
    files = os.listdir(image_dir)
    
    # 找第一个医学图像文件
    image_file = None
    for f in files:
        if f.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
            image_file = os.path.join(image_dir, f)
            break

    if image_file is None:
        print("💩 没找到医学图像文件，请放一张 .nii 或 .mhd 到 root/data 下")
        return

    print(f"📂 正在读取图像: {image_file}")
    img = utils.medi_imread(image_file)

    # 打印一些属性
    print(f"✅ 图像维度：{img.GetDimension()}")
    print(f"✅ 图像尺寸：{img.GetSize()}")
    print(f"✅ 像素类型：{img.GetPixelIDTypeAsString()}")
    print(f"✅ Direction: {img.GetDirection()}")
    print(f"✅ Origin: {img.GetOrigin()}")
    print(f"✅ Spacing: {img.GetSpacing()}\n")

def augmentation_2d_test(n_slices, img_dir=None):
    if img_dir is None:
        n_slices = 5
        img_size = 128
        fake_img = np.random.randint(0, 256, size=(n_slices, img_size, img_size)).astype(np.uint8)
        print(f"Origin: {fake_img.shape}")

        # 直接跳过 __init__，只取方法
        dummy = object.__new__(utils.ADNI_MRI)
        aug_2d = utils.ADNI_MRI.augmentation_2d
        # 或者 bind 方法：
        dummy.augmentation_2d = types.MethodType(aug_2d, dummy)

        aug_img = dummy.augmentation_2d(np.copy(fake_img))
        print(f"After: {aug_img.shape}")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(fake_img[0], cmap='gray')
        axs[0].set_title("Origin Slice 0")
        axs[1].imshow(aug_img[0, 0].numpy(), cmap='gray')
        axs[1].set_title("After Slice 0")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    else:
        files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii")])
    
        if not files:
            print("💩 没有找到 .nii 文件")
            return

        # 选第一个图像来测试
        file_path = os.path.join(img_dir, files[0])
        print(f"📂 加载图像：{file_path}")

        # 读取并转为 numpy 数组（SimpleITK 默认 [depth, height, width]）
        img_sitk = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img_sitk)  # shape: [slices, h, w]
        print(f"✅ 图像原始 shape: {img_arr.shape}")

        # 检查是否符合我们想要的 5 个切片
        if img_arr.shape[0] != n_slices:
            print("⚠️ 警告：切片数量不是 5, 请检查图像格式")
            return

        # 做一份副本用于测试
        fake_img = np.copy(img_arr)

        # 绑定类方法（跳过 __init__）
        dummy = object.__new__(utils.ADNI_MRI)
        aug_2d = utils.ADNI_MRI.augmentation_2d
        dummy.augmentation_2d = types.MethodType(aug_2d, dummy)

        # 增广
        aug_img = dummy.augmentation_2d(np.copy(fake_img))

        print(f"✨ 增广后 shape: {aug_img.shape}")

        # 可视化原图 vs 增广图（第0层）
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(fake_img[0], cmap='gray')
        axs[0].set_title("origin Slice 0")
        axs[1].imshow(aug_img[0, 0].numpy(), cmap='gray')
        axs[1].set_title("after Slice 0")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    generate_fake_npy_dataset(per_class_num=16)
