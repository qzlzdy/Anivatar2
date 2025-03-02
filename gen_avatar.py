import dnnlib
import legacy
import numpy as np
import os
import PIL.Image
import time
import torch

def main():
    network_pkl = "network-snapshot-004000.pkl"
    outdir = "avatars"
    print("Loading network from \"%s\"..." % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    G = G.cuda()
    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim]).cuda()
    total = 64
    for i in range(total):
        seed = time.time_ns() % (1 << 32)
        print("Generating avatar for seed %d (%d/%d)" % (seed, i + 1, total))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))
        z = z.cuda()
        img = G(z, label)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{outdir}/{i:04d}.png")

if __name__ == "__main__":
    main()
