import os
import math
import argparse

import torch

from torch.optim import Adam

from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel as DP

from torch.utils.data import DataLoader
import wandb

import torchvision.utils as vutils

from datetime import datetime

from sysbinder import SysBinderImageAutoEncoder
from data import GlobDataset
from utils import linear_warmup, cosine_anneal, sigreg
from config import load_config, save_config, flat_dict

parser = argparse.ArgumentParser()

# Model architecture and training hyperparameters live in the YAML config.
parser.add_argument('--config', default='configs/default.yaml',
                    help='YAML with model + train sections')

# Runtime-only arguments (not part of the model/training config).
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--checkpoint-path', default='checkpoint.pt.tar')
parser.add_argument('--data-path', default='data/*.png')
parser.add_argument('--log-path', default='logs/')

args = parser.parse_args()

cfg = load_config(args.config)
model_cfg, train_cfg = cfg.model, cfg.train

torch.manual_seed(args.seed)

log_dir = os.path.join(args.log_path, datetime.today().isoformat())
os.makedirs(log_dir, exist_ok=True)
save_config(cfg, os.path.join(log_dir, 'config.yaml'))
wandb.init(entity='jzeitler', project='sysbinder',
           config={**flat_dict(cfg), **vars(args)})


def visualize(image, recon_dvae, recon_tf, attns, N=8):

    # tile
    tiles = torch.cat((
        image[:N, None, :, :, :],
        recon_dvae[:N, None, :, :, :],
        recon_tf[:N, None, :, :, :],
        attns[:N, :, :, :, :]
    ), dim=1).flatten(end_dim=1)

    # grid
    grid = vutils.make_grid(tiles, nrow=(1 + 1 + 1 + model_cfg.num_slots), pad_value=0.8)

    return grid


train_dataset = GlobDataset(root=args.data_path, phase='train', img_size=model_cfg.image_size)
val_dataset = GlobDataset(root=args.data_path, phase='val', img_size=model_cfg.image_size)

train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': train_cfg.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

model = SysBinderImageAutoEncoder(model_cfg)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if train_cfg.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': train_cfg.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'image_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'image_decoder' in x[0]), 'lr': 0.0},
])
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(start_epoch, train_cfg.epochs):
    model.train()
    
    for idx, batch in enumerate(train_loader):
        global_step = epoch * train_epoch_size + idx

        tau = cosine_anneal(
            global_step,
            train_cfg.tau_start,
            train_cfg.tau_final,
            0,
            train_cfg.tau_steps)

        lr_warmup_factor_enc = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            train_cfg.lr_warmup_steps)

        lr_warmup_factor_dec = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            train_cfg.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / train_cfg.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = train_cfg.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * train_cfg.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * train_cfg.lr_dec

        batch = batch.cuda()

        optimizer.zero_grad()

        (recon_dvae, cross_entropy, mse, attns, slots_raw) = model(batch, tau)

        if train_cfg.use_dp:
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()

        loss = mse + cross_entropy

        sigreg_loss = torch.tensor(0.0, device=batch.device)
        if train_cfg.sigreg_weight > 0:
            slots_flat = slots_raw.flatten(0, 1)  # (B * num_slots, slot_size)
            sigreg_loss = sigreg(slots_flat, global_step, train_cfg.sigreg_num_slices)
            loss = loss + train_cfg.sigreg_weight * sigreg_loss

        loss.backward()

        clip_grad_norm_(model.parameters(), train_cfg.clip, 'inf')

        optimizer.step()

        with torch.no_grad():
            if idx % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, idx, train_epoch_size, loss.item(), mse.item()))

                wandb.log({
                    'train/loss': loss.item(),
                    'train/cross_entropy': cross_entropy.item(),
                    'train/mse': mse.item(),
                    'train/sigreg': sigreg_loss.item(),
                    'train/tau': tau,
                    'train/lr_dvae': optimizer.param_groups[0]['lr'],
                    'train/lr_enc': optimizer.param_groups[1]['lr'],
                    'train/lr_dec': optimizer.param_groups[2]['lr'],
                }, step=global_step)

    with torch.no_grad():
        recon_tf = (model.module if train_cfg.use_dp else model).reconstruct_autoregressive(batch[:8])
        grid = visualize(batch, recon_dvae, recon_tf, attns, N=8)
        wandb.log({'train_recons': wandb.Image(grid)}, step=(epoch + 1) * train_epoch_size)
    
    with torch.no_grad():
        model.eval()

        val_cross_entropy = 0.
        val_mse = 0.
        val_sigreg = 0.

        for idx, batch in enumerate(val_loader):
            batch = batch.cuda()

            (recon_dvae, cross_entropy, mse, attns, slots_raw) = model(batch, tau)

            if train_cfg.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            val_cross_entropy += cross_entropy.item()
            val_mse += mse.item()

            if train_cfg.sigreg_weight > 0:
                val_step_idx = (epoch * val_epoch_size + idx)
                slots_flat = slots_raw.flatten(0, 1)
                val_sigreg += sigreg(slots_flat, val_step_idx, train_cfg.sigreg_num_slices).item()

        val_cross_entropy /= val_epoch_size
        val_mse /= val_epoch_size
        val_sigreg /= val_epoch_size

        val_loss = val_mse + val_cross_entropy + train_cfg.sigreg_weight * val_sigreg

        val_step = (epoch + 1) * train_epoch_size

        wandb.log({
            'val/loss': val_loss,
            'val/cross_entropy': val_cross_entropy,
            'val/mse': val_mse,
            'val/sigreg': val_sigreg,
        }, step=val_step)

        print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch+1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.module.state_dict() if train_cfg.use_dp else model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

            if 50 <= epoch:
                recon_tf = (model.module if train_cfg.use_dp else model).reconstruct_autoregressive(batch[:8])
                grid = visualize(batch, recon_dvae, recon_tf, attns, N=8)
                wandb.log({'val_recons': wandb.Image(grid)}, step=val_step)

        wandb.log({'val/best_loss': best_val_loss}, step=val_step)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict() if train_cfg.use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

wandb.finish()
