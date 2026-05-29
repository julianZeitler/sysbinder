import os
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import wandb

from sysbinder import SysBinderImageAutoEncoder
from data import GlobDataset
from topology import preprocess, compute_topology, figures_from_cache, save_pngs, write_umap_html

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='data/*.png')
parser.add_argument('--output_path', default='activations.pt')
parser.add_argument('--load_activations', default=None, help='Skip inference, load existing activations.pt')
parser.add_argument('--figures_dir', default='topology_figures', help='Directory to save PNGs and topology cache')
parser.add_argument('--load_topology_cache', action='store_true', help='Load topology cache from figures_dir, skip ripser+umap')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--intermediate', action='store_true', help='Also run topology on intermediate iterations (default: last only)')

parser.add_argument('--wandb_run_id', default=None, help='Resume existing wandb run to log topology figures')
parser.add_argument('--wandb_project', default='sysbinder')
parser.add_argument('--wandb_entity', default='jzeitler')
parser.add_argument('--pca_variance', type=float, default=0.95, help='Variance ratio retained by PCA before homology')
parser.add_argument('--n_landmarks', type=int, default=500, help='Maxmin landmarks subsampled before PCA/ripser')
parser.add_argument('--umap_n_landmarks', type=int, default=2000, help='Maxmin landmarks subsampled for UMAP embedding')
parser.add_argument('--umap_image_hover', action='store_true', help='Save slot attention images and embed paths in UMAP HTML for hover preview')

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=4)
parser.add_argument('--num_blocks', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=512)
parser.add_argument('--slot_size', type=int, default=2048)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_prototypes', type=int, default=64)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_layers', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

args = parser.parse_args()

torch.manual_seed(args.seed)

if args.wandb_run_id is not None:
    wandb.init(entity=args.wandb_entity, project=args.wandb_project,
               id=args.wandb_run_id, resume='must')

# ── activations ──────────────────────────────────────────────────────────────

slot_img_dir = None

if args.load_topology_cache:
    per_iteration = []  # not needed, topology cache has everything
elif args.load_activations is not None:
    print(f'Loading activations from {args.load_activations}')
    saved = torch.load(args.load_activations, map_location='cpu')
    per_iteration = saved['per_iteration']
    args.num_iterations = len(per_iteration)
    args.num_blocks = saved['args'].get('num_blocks', args.num_blocks)
    print(f'  per_iteration: {len(per_iteration)} x {per_iteration[0].shape}')
    slot_img_dir = saved.get('slot_img_dir')
    if args.umap_image_hover and slot_img_dir is None:
        print('Warning: activations.pt has no slot images. Re-run without --load_activations to generate them.')
else:
    model = SysBinderImageAutoEncoder(args)

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint_path}')

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()

    dataset = GlobDataset(root=args.data_path, phase='test' if args.test_only else 'all', img_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    all_slot_histories = []
    all_attn_vis = []
    all_images = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda()

            emb = model.image_encoder.cnn(batch)
            emb = model.image_encoder.pos(emb)
            H_enc, W_enc = emb.shape[-2:]
            B = batch.shape[0]

            emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
            emb_set = model.image_encoder.mlp(model.image_encoder.layer_norm(emb_set))
            emb_set = emb_set.reshape(B, H_enc * W_enc, args.d_model)

            _, attn_vis, slot_history = model.image_encoder.sysbinder(emb_set, return_intermediates=True)
            # attn_vis: (B, H_enc*W_enc, num_slots) — from last sysbinder iteration

            all_slot_histories.append([s.cpu() for s in slot_history])
            all_attn_vis.append(attn_vis.cpu())
            all_images.append(batch.cpu())

    per_iteration = [
        torch.cat([batch[i] for batch in all_slot_histories], dim=0)
        for i in range(args.num_iterations)
    ]

    # ── save slot attention visualizations ───────────────────────────────────
    if args.umap_image_hover:
        slot_img_dir = os.path.abspath(args.output_path.replace('.pt', '_slot_images'))
        os.makedirs(slot_img_dir, exist_ok=True)
        print(f'Saving slot attention images to {slot_img_dir}...')

        H, W = args.image_size, args.image_size
        global_idx = 0
        for attn_batch, img_batch in zip(all_attn_vis, all_images):
            B = img_batch.shape[0]
            # attn_batch: (B, H_enc*W_enc, num_slots) → (B, num_slots, H_enc*W_enc)
            attn = attn_batch.transpose(1, 2)
            # reshape to spatial: (B, num_slots, H_enc, W_enc)
            H_enc_b = W_enc_b = int(attn.shape[-1] ** 0.5)
            attn = attn.reshape(B, args.num_slots, H_enc_b, W_enc_b)
            # upsample to full image size: (B*num_slots, 1, H, W)
            attn_up = F.interpolate(
                attn.flatten(0, 1).unsqueeze(1),
                size=(H, W), mode='bilinear', align_corners=False
            ).reshape(B, args.num_slots, 1, H, W)
            # blend: image * attn + (1 - attn) white background
            vis = img_batch.unsqueeze(1) * attn_up + (1.0 - attn_up)  # (B, num_slots, C, H, W)

            for b in range(B):
                for s in range(args.num_slots):
                    save_image(vis[b, s], os.path.join(slot_img_dir, f'{global_idx:06d}_{s}.png'))
                global_idx += 1

        print(f'  Saved {global_idx * args.num_slots} slot images.')

    torch.save({
        'per_iteration': per_iteration,
        'args': vars(args),
        'slot_img_dir': slot_img_dir,
    }, args.output_path)

    print(f'Saved activations for {len(dataset)} images to {args.output_path}')
    print(f'  per_iteration: {len(per_iteration)} x {per_iteration[0].shape}')

# ── topology ──────────────────────────────────────────────────────────────────

os.makedirs(args.figures_dir, exist_ok=True)
cache_path = os.path.join(args.figures_dir, 'topology_cache.pkl')

iters = range(len(per_iteration)) if args.intermediate else [len(per_iteration) - 1]

if args.load_topology_cache:
    print(f'Loading topology cache from {cache_path}')
    with open(cache_path, 'rb') as f:
        topology_cache = pickle.load(f)
else:
    topology_cache = {}
    for i in iters:
        print(f'Topology: iteration {i+1}/{len(per_iteration)}...')
        N, num_slots, slot_size = per_iteration[i].shape
        d_block = slot_size // args.num_blocks
        arr = per_iteration[i].numpy()
        for j in range(args.num_blocks):
            print(f'  block {j+1}/{args.num_blocks}', flush=True)
            block = arr[:, :, j * d_block:(j + 1) * d_block]
            data = block.reshape(N * num_slots, d_block)
            pcs = preprocess(data, variance_ratio=args.pca_variance, n_landmarks=args.n_landmarks)
            if args.umap_image_hover and slot_img_dir:
                img_paths = [
                    os.path.join(slot_img_dir, f'{n:06d}_{s}.png')
                    for n in range(N) for s in range(num_slots)
                ]
            else:
                img_paths = None
            entry = compute_topology(pcs, umap_n_landmarks=args.umap_n_landmarks, raw_data=data, image_paths=img_paths)
            topology_cache[f'iter{i}/block{j}'] = entry

    with open(cache_path, 'wb') as f:
        pickle.dump(topology_cache, f)
    print(f'Topology cache saved to {cache_path}')

# ── figures & logging ─────────────────────────────────────────────────────────

log_dict = {}
for key, entry in topology_cache.items():
    prefix = f'topology/{key}'
    if args.wandb_run_id is not None:
        barcode, diagram, umap_2d, umap_3d = figures_from_cache(entry)
        log_dict[f'{prefix}/barcode'] = wandb.Plotly(barcode)
        log_dict[f'{prefix}/diagram'] = wandb.Plotly(diagram)
        log_dict[f'{prefix}/umap']    = wandb.Plotly(umap_3d)
    else:
        base = os.path.join(args.figures_dir, key.replace("/", "__"))
        save_pngs(entry, base)
        _, _, _, umap_3d = figures_from_cache(entry)
        write_umap_html(umap_3d, base + '__umap.html')
    log_dict.update({f'{prefix}/{k}': v for k, v in entry['metrics'].items()})

if args.wandb_run_id is not None:
    wandb.log(log_dict)
    wandb.finish()
    print('Topology figures logged to wandb.')
else:
    for k, v in log_dict.items():
        if not hasattr(v, 'write_image'):
            print(f'  {k}: {v}')
    print(f'Figures saved to {args.figures_dir}/')
