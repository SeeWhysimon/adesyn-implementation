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
    print(f"🚧 Generating data... {per_class_num} per class, {per_class_num * len(classes)} in total.")

    for cls in classes:
        class_dir = os.path.join(save_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(per_class_num):
            fake_data = np.random.randint(
                0, 256, size=(img_depth, img_size, img_size)
            ).astype(np.uint8)

            out_path = os.path.join(class_dir, f"fake_{cls}_{i:03d}.npy")
            np.save(out_path, fake_data)

            print(f"✅ Saved: {out_path}")

    print(f"\n🎉 Finished generation, {len(classes)} classes * {per_class_num} = {per_class_num * len(classes)} in total.")

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
        print("💩 No medi image detected, please place a .nii or .mhd file to data directory")
        return

    print(f"📂 Reading image: {image_file}...")
    img = utils.medi_imread(image_file)

    # 打印一些属性
    print(f"✅ Dimension: {img.GetDimension()}")
    print(f"✅ Size: {img.GetSize()}")
    print(f"✅ Pixel type: {img.GetPixelIDTypeAsString()}")
    print(f"✅ Direction: {img.GetDirection()}")
    print(f"✅ Origin: {img.GetOrigin()}")
    print(f"✅ Spacing: {img.GetSpacing()}\n")

def ADNI_MRI_test(img_dir="../data/NP_3D/setA", nserial=3, mode="train", visualize=False):
    dataset = utils.ADNI_MRI(image_dir=img_dir, nserial=nserial, mode=mode)
    print(f"\n📦 {len(dataset)} images loaded.")

    # 读取第一张图像
    aug2d_img, label_out, aug3d_img = dataset[0]

    # 获取原始图像维度
    raw_path = dataset.dataset[0][0]  # .npy 文件路径
    raw_data = np.load(raw_path)
    print(f"📐 Original image shape: {raw_data.shape}")  # [depth, H, W]

    print(f"🎨 After 2D augmentation shape: {aug2d_img.shape}")   # [nserial, 1, H, W]
    print(f"🧱 After 3D augmentation shape: {aug3d_img.shape}")   # [nserial, H, W]（没加 channel）
    print(f"🧾 标签 shape: {label_out.shape}")             # [nserial, 1]

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
