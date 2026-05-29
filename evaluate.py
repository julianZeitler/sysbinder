import os
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from sysbinder import SysBinderImageAutoEncoder
from data import GlobDataset
from topology import preprocess, compute_topology, figures_from_cache, save_pngs

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
parser.add_argument('--last_iter_only', action='store_true', help='Only run topology on the last iteration')

parser.add_argument('--wandb_run_id', default=None, help='Resume existing wandb run to log topology figures')
parser.add_argument('--wandb_project', default='sysbinder')
parser.add_argument('--wandb_entity', default='jzeitler')
parser.add_argument('--pca_variance', type=float, default=0.95, help='Variance ratio retained by PCA before homology')
parser.add_argument('--n_landmarks', type=int, default=500, help='Maxmin landmarks subsampled before PCA/ripser')
parser.add_argument('--umap_n_landmarks', type=int, default=2000, help='Maxmin landmarks subsampled for UMAP embedding')

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

if args.load_topology_cache:
    per_iteration = []  # not needed, topology cache has everything
elif args.load_activations is not None:
    print(f'Loading activations from {args.load_activations}')
    saved = torch.load(args.load_activations, map_location='cpu')
    per_iteration = saved['per_iteration']
    args.num_iterations = len(per_iteration)
    args.num_blocks = saved['args'].get('num_blocks', args.num_blocks)
    print(f'  per_iteration: {len(per_iteration)} x {per_iteration[0].shape}')
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

            _, _, slot_history = model.image_encoder.sysbinder(emb_set, return_intermediates=True)

            all_slot_histories.append([s.cpu() for s in slot_history])

    per_iteration = [
        torch.cat([batch[i] for batch in all_slot_histories], dim=0)
        for i in range(args.num_iterations)
    ]

    torch.save({
        'per_iteration': per_iteration,
        'args': vars(args),
    }, args.output_path)

    print(f'Saved activations for {len(dataset)} images to {args.output_path}')
    print(f'  per_iteration: {len(per_iteration)} x {per_iteration[0].shape}')

# ── topology ──────────────────────────────────────────────────────────────────

os.makedirs(args.figures_dir, exist_ok=True)
cache_path = os.path.join(args.figures_dir, 'topology_cache.pkl')

iters = [len(per_iteration) - 1] if args.last_iter_only else range(len(per_iteration))

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
            data = block.reshape(N, -1)
            pcs = preprocess(data, variance_ratio=args.pca_variance, n_landmarks=args.n_landmarks)
            entry = compute_topology(pcs, umap_n_landmarks=args.umap_n_landmarks, raw_data=data)
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
        umap_3d.write_html(base + '__umap.html')
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
