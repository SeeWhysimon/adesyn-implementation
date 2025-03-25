# Implementation for Conditional-GAN-with-an-Attention-based-Generator-and-a-3D-Discriminator-for-3D-Medical-Image

This repository is an implementation for the github repository (ADESyn)[https://github.com/EuijinMisp/ADESyn].

## Package Installation
Run `pip install -r requirements.txt` to install basic dependecies. Furthermore, run `pip install -e .` to install local `adesyn` package.

## Data preparation
Create `data` directory at project root. Then create directory `setA` and `setC` in `data`. If you want to add data manually, create `AD`, `NM` and `MCI` sub-directory in each set respectively. Place your images in each directory in `.npy` format. Make sure the number of images in each sub-directory are same.

Fake data generation is supported by following steps below.

1. Direct to `project_root/scripts`.
2. Go through file `utils_test.py` and select the `generate_fake_npy_dataset`.
3. Decide parameters for your generation and run `python utils_test.py`.

## Network training
To train the network, follow the instruction in `project_root/adesyn/main.py`.