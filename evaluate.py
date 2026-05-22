import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from sysbinder import SysBinderImageAutoEncoder
from data import GlobDataset
from topology import preprocess, persistent_homology

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='data/*.png')
parser.add_argument('--output_path', default='activations.pt')
parser.add_argument('--test_only', action='store_true')

parser.add_argument('--wandb_run_id', default=None, help='Resume existing wandb run to log topology figures')
parser.add_argument('--wandb_project', default='sysbinder')
parser.add_argument('--wandb_entity', default='jzeitler')
parser.add_argument('--pca_variance', type=float, default=0.95, help='Variance ratio retained by PCA before homology')

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

# load model
model = SysBinderImageAutoEncoder(args)

if not os.path.isfile(args.checkpoint_path):
    raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint_path}')

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model = model.cuda()
model.eval()

# dataset
dataset = GlobDataset(root=args.data_path, phase='test' if args.test_only else 'all', img_size=args.image_size)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

# collect activations
# all_slots[i] = (B, num_slots, slot_size) tensor after iteration i, concatenated over batches
all_final_slots = []
all_slot_histories = []  # list of lists: [batch][iteration] -> (B, num_slots, slot_size)

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

        slots, _, slot_history = model.image_encoder.sysbinder(emb_set, return_intermediates=True)

        all_final_slots.append(slots.cpu())
        all_slot_histories.append([s.cpu() for s in slot_history])

# stack final slots: (N_total, num_slots, slot_size)
all_final_slots = torch.cat(all_final_slots, dim=0)

# stack per-iteration: list of (N_total, num_slots, slot_size), one per iteration
per_iteration = [
    torch.cat([batch[i] for batch in all_slot_histories], dim=0)
    for i in range(args.num_iterations)
]

torch.save({
    'final_slots': all_final_slots,       # (N, num_slots, slot_size)
    'per_iteration': per_iteration,        # list[num_iterations] of (N, num_slots, slot_size)
    'args': vars(args),
}, args.output_path)

print(f'Saved activations for {len(dataset)} images to {args.output_path}')
print(f'  final_slots: {all_final_slots.shape}')
print(f'  per_iteration: {len(per_iteration)} x {per_iteration[0].shape}')

# topology analysis — per block
# slots shape: (N, num_slots, slot_size), block j at [..., j*d_block:(j+1)*d_block]
def _topology_per_block(slots_tensor, prefix, log_dict):
    N, num_slots, slot_size = slots_tensor.shape
    d_block = slot_size // args.num_blocks
    arr = slots_tensor.numpy()
    for j in range(args.num_blocks):
        block = arr[:, :, j * d_block:(j + 1) * d_block]  # (N, num_slots, d_block)
        data = block.reshape(N, -1)                         # (N, num_slots * d_block)
        pcs = preprocess(data, variance_ratio=args.pca_variance)
        fig, metrics = persistent_homology(pcs)
        log_dict[f'{prefix}/block{j}/barcode'] = wandb.Plotly(fig)
        log_dict.update({f'{prefix}/block{j}/{k}': v for k, v in metrics.items()})

if args.wandb_run_id is not None:
    log_dict = {}
    for i, slots in enumerate(per_iteration):
        _topology_per_block(slots, f'topology/iter{i}', log_dict)
    _topology_per_block(all_final_slots, 'topology/final', log_dict)

    wandb.log(log_dict)
    wandb.finish()
    print('Topology figures logged to wandb.')
