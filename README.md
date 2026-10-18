# MIDDiffRec

PyTorch implementation of **Multi-Intent Distribution-Driven Conditional and Guided Diffusion with Contrastive Learning for Sequential Recommendation**.

MIDDiffRec models each user's preferences as a distribution over intent clusters. It combines multi-intent distribution representations with long-term and short-term preferences to guide diffusion over target item embeddings. Condition Diffusion and Guidance Diffusion share a denoising network, and the generated augmented views are aligned with the original sequence representations through contrastive learning.

## Requirements

- Python 3.9 or later, with compatible package versions
- PyTorch with CUDA support
- GPU-enabled Faiss
- NumPy, SciPy, and tqdm

## Code Structure

| File | Purpose |
| --- | --- |
| `main.py` | Command-line arguments, data loading, training, and evaluation |
| `models.py` | Diffusion model, intent representation, and clustering |
| `bsarec_complete.py` | BSARec sequence encoder required by `models.py` |
| `modules.py` | Neural network components |
| `trainer.py` | Training objectives, intent-distribution construction, and evaluation |
| `datasets.py` | Sequence preprocessing and dataset classes |
| `utils.py` | Metrics, early stopping, and utility functions |

## Datasets

Place the processed interaction sequences in `datasets/` using the following names:

| Dataset | `--data_name` | Sequence file |
| --- | --- | --- |
| Beauty | `Beauty` | `datasets/Beauty.txt` |
| Sports | `Sports_and_Outdoors` | `datasets/Sports_and_Outdoors.txt` |
| Toys | `Toys_and_Games` | `datasets/Toys_and_Games.txt` |
| Video | `Video` | `datasets/Video.txt` |
| MovieLens-1M | `ml-1m` | `datasets/ml-1m.txt` |

Each line contains a user ID followed by chronologically ordered item IDs, separated by spaces:

```text
user_id item_id_1 item_id_2 ... item_id_n
```

The program generates `<data_name>_s.txt` for subsequence training if that file is missing. Dataset files and model checkpoints are excluded from Git by `.gitignore` and must be prepared locally.

## Training

Run commands from the repository root. Use the following command to train on Beauty with reverse-diffusion augmentation and DDIM sampling.

```bash
python main.py \
  --data_name Beauty \
  --epochs 100 \
  --batch_size 256 \
  --timesteps 50 \
  --rec_weight 1.0 \
  --diff_weight 1.0 \
  --diffusion_rec_weight 0.01 \
  --tau 12 \
  --p 0.2 \
  --intent_num 32 \
  --cl_use_reverse_process \
  --cl_weight 0.2 \
  --w 1 \
  --use_ddim
```

Training creates the output directory, saves the best checkpoint according to validation NDCG@20, and evaluates that checkpoint on the test set. With the default `--model_name InDiRec` and `--model_idx 0`, the output files are:

```text
output/InDiRec-Beauty-0.pt
output/InDiRec-Beauty-0.txt
```

The `--cl_use_reverse_process` option generates contrastive views through reverse diffusion. When this option is omitted, the current implementation reuses its single-step denoising prediction for contrastive learning.

## Evaluation

After training, evaluate the saved checkpoint with the same model and dataset configuration:

```bash
python main.py \
  --data_name Beauty \
  --epochs 100 \
  --batch_size 256 \
  --timesteps 50 \
  --rec_weight 1.0 \
  --diff_weight 1.0 \
  --diffusion_rec_weight 0.01 \
  --tau 12 \
  --p 0.2 \
  --intent_num 32 \
  --cl_use_reverse_process \
  --cl_weight 0.2 \
  --w 1 \
  --use_ddim \
  --eval_only
```

The checkpoint path is constructed as `<output_dir>/<model_name>-<data_name>-<model_idx>.pt`. Supply an existing checkpoint and keep its architecture and preprocessing settings unchanged. Pretrained checkpoints are not bundled with this repository.

The code retains the internal names `InDiRec` and `InDiRecTrainer` from its implementation history. `--model_name MIDDiffRec` controls output filenames. To load an older checkpoint named `InDiRec-Beauty-91.pt`, use `--model_name InDiRec --data_name Beauty --model_idx 91` together with that checkpoint's original configuration.

For all supported arguments, run:

```bash
python main.py --help
```

## Acknowledgments

This implementation is developed from [InDiRec: Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation](https://doi.org/10.1145/3726302.3730010) and uses BSARec as its sequence encoder. The upstream implementation also builds on [ICLRec](https://github.com/salesforce/ICLRec) and [DreamRec](https://github.com/YangZhengyi98/DreamRec). We thank the authors for sharing their implementations.

## Contact

For questions, please contact [koizcr@stu.cqu.edu.cn](mailto:koizcr@stu.cqu.edu.cn).
