import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sysbinder import SysBinderImageAutoEncoder
from data import GlobDataset
from config import load_config

parser = argparse.ArgumentParser(
    description='Plot the softmax distribution over block prototype memories for a single image.')
parser.add_argument('--config', default='configs/default.yaml',
                    help='YAML with the model section (must match the checkpoint)')
parser.add_argument('--checkpoint-path', default='checkpoint.pt.tar')
parser.add_argument('--data-path', default='data/*.png')
parser.add_argument('--image-index', type=int, default=0,
                    help='Index into the (sorted) dataset glob to inspect')
parser.add_argument('--output-path', default=None,
                    help='PNG path (default: <checkpoint dir>/memory_attention.png)')
args = parser.parse_args()

cfg = load_config(args.config)
model_cfg = cfg.model

model = SysBinderImageAutoEncoder(model_cfg)
if not os.path.isfile(args.checkpoint_path):
    raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint_path}')
checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# single image
dataset = GlobDataset(root=args.data_path, phase='all', img_size=model_cfg.image_size)
image = dataset[args.image_index].unsqueeze(0)  # 1, C, H, W

# capture the softmax over memories across the Hopfield retrieval iterations of
# the FINAL sysbinder iteration
attn_mod = model.image_encoder.sysbinder.prototype_memory.attn
num_retrieval_iters = model.image_encoder.sysbinder.prototype_memory.num_retrieval_iters
attn_mod.store_attn = True
attn_mod.attn_history = []
with torch.no_grad():
    model.encode(image)  # runs the encoder + sysbinder; attn captured every retrieval step
attn_mod.store_attn = False

# attn_history has num_sysbinder_iters * num_retrieval_iters entries; keep the last
# block (the final sysbinder iteration's retrieval steps)
hist = attn_mod.attn_history[-num_retrieval_iters:]
attn = torch.stack([h[0] for h in hist]).cpu().numpy()  # R, num_blocks, num_slots, num_prototypes
R, num_blocks, num_slots, num_prototypes = attn.shape
ks = range(num_prototypes)
cmap = plt.get_cmap('Blues')
# light blue (early) -> dark blue (late); single-iter case still gets a defined color
colors = [cmap(0.35 + 0.65 * (r / max(R - 1, 1))) for r in range(R)]

out = args.output_path or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint_path)),
                                       'memory_attention.png')
base, ext = os.path.splitext(out)
ext = ext or '.png'

for s in range(num_slots):
    fig, axes = plt.subplots(num_blocks, 1, figsize=(9, 1.6 * num_blocks),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for b in range(num_blocks):
        ax = axes[b]
        for r in range(R):
            ax.plot(ks, attn[r, b, s], color=colors[r], marker='.', linewidth=1.0,
                    label=f'iter {r}' if b == 0 else None)
        # prototypes needed (largest→smallest) to reach 90% of the mass, final iteration
        cumsum = np.cumsum(np.sort(attn[-1, b, s])[::-1])
        n90 = int(np.searchsorted(cumsum, 0.9) + 1)
        ax.set_ylabel(f'block {b}', fontsize=8)
        ax.set_title(f'{n90}/{num_prototypes} prototypes for 90% (final iter)',
                     fontsize=7, loc='right', pad=2)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel('memory index $k$')
    if R > 1:
        axes[0].legend(fontsize=6, ncol=R, loc='upper right')
    fig.suptitle(f'Memory attention — image {args.image_index}, slot {s}, '
                 f'{num_blocks} blocks, {num_prototypes} prototypes, {R} retrieval iters')
    fig.supylabel('softmax weight')
    plt.tight_layout()
    path = f'{base}_slot{s}{ext}'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')
