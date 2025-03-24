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

def ADNI_MRI_test(img_dir="../data/NP_3D/setA", nserial=3, mode="train", visualize=False):
    dataset = utils.ADNI_MRI(image_dir=img_dir, nserial=nserial, mode=mode)
    print(f"\n📦 总共加载图像数量: {len(dataset)}")

    # 读取第一张图像
    aug2d_img, label_out, aug3d_img = dataset[0]

    # 获取原始图像维度
    raw_path = dataset.dataset[0][0]  # .npy 文件路径
    raw_data = np.load(raw_path)
    print(f"📐 原始图像 shape: {raw_data.shape}")  # [depth, H, W]

    print(f"🎨 增强后 2D 图像 shape: {aug2d_img.shape}")   # [nserial, 1, H, W]
    print(f"🧾 标签 shape: {label_out.shape}")             # [nserial, 1]
    print(f"🧱 增强后 3D 图像 shape: {aug3d_img.shape}")   # [nserial, H, W]（没加 channel）

    if visualize:
        # 可视化对比原图 vs 增强后图（第0层）
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(raw_data[50], cmap='gray')
        axs[0].set_title("origin Slice 50")

        axs[1].imshow(aug2d_img[0, 0].numpy(), cmap='gray')
        axs[1].set_title("2D aud Slice 0")

        axs[2].imshow(aug3d_img[0].numpy(), cmap='gray')
        axs[2].set_title("3D aug Slice 0")

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    generate_fake_npy_dataset(save_dir="../data/NP_3D/setC")
