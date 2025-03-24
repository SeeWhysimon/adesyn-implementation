import types
import numpy as np
import torch
import matplotlib.pyplot as plt
from adesyn.utils import ADNI_MRI

def visualize_augmentation():
    n_slices = 5
    img_size = 128
    fake_img = np.random.randint(0, 256, size=(n_slices, img_size, img_size)).astype(np.uint8)
    print(f"Origin: {fake_img.shape}")

    # 直接跳过 __init__，只取方法
    dummy = object.__new__(ADNI_MRI)
    aug_2d = ADNI_MRI.augmentation_2d
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

if __name__ == "__main__":
    visualize_augmentation()
