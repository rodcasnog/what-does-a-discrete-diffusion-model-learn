# What Does a Discrete Diffusion Model Learn? One Reverse Process in Three Coordinates

Code for [arXiv:2607.05381](https://arxiv.org/abs/2607.05381). A discrete diffusion model learns the reverse noising process, which the network parameterizes in three possible coordinates: denoiser, bridge plug-in, and concrete score. This repository trains a head in any coordinate under masked, GIDD or uniform noise, converts between the coordinates, evaluates the NELBO, and samples with the ancestral or the re-corrupt sampler, on OpenWebText and four logic tasks: Sudoku, Boxoban, mazes and graph coloring.

## Install

`uv sync` targets Linux x86_64 with CUDA 13 GPUs; on macOS it installs a CPU build of JAX. Training ran with `XLA_FLAGS="--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true"` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, under which OpenWebText at batch 512 needs about 87 GB of GPU memory: one 96 GB GPU, or the batch split over several. The maze model at batch 512 needs four 24 GB GPUs or larger ones.

## Data

Run every command from the repository root. Each prep script writes `.npy` arrays where the configs read them. Graph-coloring instances are generated on the fly.

```bash
uv run scripts/prep_owt.py --out data/openwebtext-gpt2-len256
uv run scripts/prep_sudoku.py --out data/sudoku-extreme
uv run scripts/prep_boxoban.py --out data/boxoban
uv run scripts/prep_maze.py --out data/maze-40x40
```

## Train

| Config | Model |
|---|---|
| `paper/owt/masked_{cavity,score}` | masked diffusion, where the denoiser and cavity heads coincide |
| `paper/owt/gidd01_{denoiser,cavity,score}`, `paper/owt/gidd02_*` | GIDD with lambda = 0.1 and 0.2 |
| `paper/owt/uniform_{denoiser,cavity,score}` | uniform diffusion |
| `sudoku`, `boxoban`, `maze`, `graphcolor` | uniform diffusion, cavity head |

```bash
uv run python -m discrete_diffusion.train --config-name paper/owt/masked_cavity hydra.run.dir=runs/masked_cavity
uv run python -m discrete_diffusion.train --config-name sudoku training.seed=0 hydra.run.dir=runs/sudoku
```

A run directory holds the composed config, `metrics.jsonl` and `checkpoints/<step>`; rerunning its command resumes from the latest checkpoint. Each batch is split over the visible GPUs of one host, with no gradient accumulation. `wandb.mode=online` also logs the metrics to Weights & Biases.

## Sample and evaluate

`sample`, `eval.elbo` and `eval.solve` read `checkpoint_dir=<run>/checkpoints/<step>`, `eval.heads` a list of them as `views` and `eval.text` the samples; each writes to its Hydra output directory. `read_as=<coordinate>` reads a head in a coordinate it was not trained in, without converting it.

```bash
# Text samples, then GenPPL, word entropy and MAUVE under GPT-2 Large
uv run python -m discrete_diffusion.sample checkpoint_dir=runs/masked_cavity/checkpoints/500000 hydra.run.dir=outputs/masked_cavity
uv run python -m discrete_diffusion.eval.text samples=outputs/masked_cavity/samples.json
# Test NELBO of a cavity head misread as a denoiser
uv run python -m discrete_diffusion.eval.elbo checkpoint_dir=runs/gidd01_cavity/checkpoints/500000 read_as=denoiser
# Agreement of one kernel's heads; stat=sensitivity redraws one token per sequence instead
U=runs/uniform_cavity/checkpoints/500000
uv run python -m discrete_diffusion.eval.heads "views=[$U,$U:denoiser,runs/uniform_denoiser/checkpoints/500000]"
# Test NELBO of a logic model, then its solve rates for both samplers, native and misread, at several NFE
uv run python -m discrete_diffusion.eval.elbo checkpoint_dir=runs/sudoku/checkpoints/1000000 n=64000 batch_size=256
uv run python -m discrete_diffusion.eval.solve -m checkpoint_dir=runs/sudoku/checkpoints/1000000 sampler=ancestral,re_corrupt read_as=null,denoiser nfe=16,64,256
```

## Citation

```bibtex
@misc{casadonoguerales2026discretediffusion,
  title         = {What Does a Discrete Diffusion Model Learn? One Reverse
                   Process in Three Coordinates},
  author        = {Casado Noguerales, Rodrigo and Sch{\"o}lkopf, Bernhard
                   and Hofmann, Thomas and Raoufi, Aran},
  year          = {2026},
  eprint        = {2607.05381},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.05381},
}
```

## License

Apache-2.0; see `LICENSE`.
