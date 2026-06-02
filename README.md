
# Neural Systematic Binder
*ICLR 2023*

#### [[arXiv](https://arxiv.org/abs/2211.01177)] [[project](https://sites.google.com/view/neural-systematic-binder)] [[datasets](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing)] [[openreview](https://openreview.net/forum?id=ZPHE4fht19t)]

This is the **official PyTorch implementation** of _Neural Systematic Binder_.

<img src="https://i.imgur.com/hqwcCpU.png">

### Authors
Gautam Singh and Yeongbin Kim and Sungjin Ahn

### Datasets
The datasets tested in the paper (CLEVR-Easy, CLEVR-Hard, and CLEVR-Tex) can be downloaded via this [link](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing).

### Configuration
Model architecture and training hyperparameters live in a YAML file with two
sections, `model` (architecture — must match a checkpoint) and `train`
(optimization). See `configs/default.yaml`. To run an experiment, copy it and
edit the fields you want:
```bash
cp configs/default.yaml configs/my_run.yaml
python train.py --config configs/my_run.yaml
```
Runtime concerns (data paths, seed, wandb, checkpointing) remain CLI flags.
`train.py` writes the resolved config to `<log_path>/<timestamp>/config.yaml`
next to `best_model.pt`, so evaluation can reuse the exact architecture with no
flags to retype:
```bash
python evaluate.py --config logs/<run>/config.yaml \
                   --checkpoint-path logs/<run>/checkpoint.pt.tar
```

`evaluate.py` writes its outputs (activations, topology figures + cache, slot
images, and a copy of `config.yaml`) into a single directory. By default this is
the **checkpoint's own directory** (the training run's `logs/<run>/`), so eval
artifacts sit beside the model they came from. Pass `--output-path <dir>` to
write elsewhere. (`--wandb-run-id` only controls wandb logging, not where files
land.)

### Training
To train the model with the default config, simply execute:
```bash
python train.py
```
Use `--data-path` to point to the set of images via a glob pattern, and
`--config` to select a YAML config (defaults to `configs/default.yaml`).

### Outputs
The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory that was provided in the training argument `--log_path`. These logs contain the training loss curves and visualizations of reconstructions and object attention maps.

### Packages Required
The following packages may need to be installed first.
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://pypi.org/project/tensorboard/) for logging.

### Evaluation
The evaluation scripts are provided in branch `evaluate`.

### Citation
```
@inproceedings{
      singh2023sysbinder,
      title={Neural Systematic Binder},
      author={Gautam Singh and Yeongbin Kim and Sungjin Ahn},
      booktitle={International Conference on Learning Representations},
      year={2023},
      url={https://openreview.net/forum?id=ZPHE4fht19t}
}
```