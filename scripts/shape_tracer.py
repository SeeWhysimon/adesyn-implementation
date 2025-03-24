import torch
from adesyn.network import Generator, Discriminator_2D, Discriminator_3D
import torch.nn as nn
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='g', choices=['g', 'd2d', 'd3d'])
    return parser.parse_args()

def register_shape_hooks(model):
    hooks = []

    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        module_idx = len(hooks)
        m_key = f"{module_idx:02d}-{class_name}"
        in_shape = [tuple(i.shape) for i in input]
        out_shape = output.shape if isinstance(output, torch.Tensor) else str(output)
        print(f"{m_key:20} | Input: {in_shape} | Output: {out_shape}")

    for module in model.modules():
        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and module != model:
            hooks.append(module.register_forward_hook(hook_fn))

    return hooks


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == "g":
        model = Generator().to(device)
        dummy_input = (torch.randn(1, 1, 192, 192).to(device), torch.randn(1, 1).to(device))
    elif args.model == "d2d":
        model = Discriminator_2D().to(device)
        dummy_input = (torch.randn(1, 1, 192, 192).to(device), )
    elif args.model == "d3d":
        model = Discriminator_3D().to(device)
        dummy_input = (torch.randn(1, 1, 6, 192, 192).to(device), )

    model.eval()
    print(f"\n=== Registering Shape Hooks for {args.model} ===")
    hooks = register_shape_hooks(model)

    print("\n=== Forward Pass ===")
    with torch.no_grad():
        model(*dummy_input)

    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
