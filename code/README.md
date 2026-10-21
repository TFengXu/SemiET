# SemiET

Official implementation of **SemiET: Semi-supervised fundus enhancement with efficient Transformers**
(Biomedical Signal Processing and Control, 2026; article reference BSPC 111659).

SemiET enhances retinal fundus images with a semi-supervised framework built on an
efficient Transformer backbone (Restormer). It combines:

- a **supervised branch** trained on synthetically degraded high-quality images
  (pixel reconstruction + LSGAN), and
- an **unsupervised branch** trained on real low-quality images
  (patch-wise contrastive loss + identity loss + LSGAN),

so that the model benefits from paired synthetic data and unpaired real data at the same time.

## News

- 2026: accepted by *Biomedical Signal Processing and Control*.

## Repository structure

```
SemiET/
├── main.py / train.py / test.py / option.py   # entry points and options
├── isecret/                                   # model, losses, datasets, utils
├── tools/                                     # degradation, PSNR/SSIM/LPIPS, FLOPs
├── scripts/                                   # training / inference / ablation
├── downstream/                                # how to run the downstream evaluations
├── docs/                                      # data, reproduction, results
├── assets/                                    # figures
├── train.csv / val.csv / test.csv             # EyeQ split lists
└── requirements.txt
```

## Installation

```bash
conda create -n semiet python=3.8 -y
conda activate semiet
# install the PyTorch build matching your CUDA driver, e.g.
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data preparation

See [docs/data.md](docs/data.md). In short:

1. download the EyeQ dataset and split it into `train/val/test`;
2. put the good-quality crops under `data/eyeq/<split>/crop_good`;
3. generate synthetic low-quality pairs with `tools/degrade_eyeq.py`;
4. the resulting layout is `data/eyeq/<split>/{crop_good,degrade_good,crop_usable,mask_good}`.

## Pretrained weights

Download `b2g_gen_best.pt` and `config.yaml` from the GitHub **Releases** page and place them at:

```
experiments/eyeq/512/semiet/b2g_gen_best.pt
experiments/eyeq/512/semiet/config/config.yaml
```

The testing entry point loads `best` automatically (no extra flag needed). If `config.yaml`
is missing the script falls back to the command-line arguments.

## Inference and evaluation

```bash
# enhance a folder of low-quality images
bash scripts/test_semiet.sh data/eyeq/test/degrade_good

# PSNR / SSIM
python tools/evaluate.py \
    --test_dir data/eyeq/test/degrade_semiet_enhance \
    --gt_dir data/eyeq/test/crop_good \
    --mask_dir data/eyeq/test/mask_good

# LPIPS
python tools/evaluate_lpips.py \
    --gt_dir data/eyeq/test/crop_good \
    --mask_dir data/eyeq/test/mask_good \
    --base_test_dir data/eyeq/test \
    --variants semiet
```

## Training

```bash
bash scripts/train_semiet.sh
```

The released checkpoint uses: 512 input, 256 random crop, Restormer backbone,
`lambda_rec=100`, `lambda_icc=1`, `lambda_idt=1`, `lambda_gan=1`,
AdamBelief, lr `5e-5`, batch size 8, 200 epochs, cosine schedule.

## Ablation study

```bash
bash scripts/ablation/train_ablation.sh abl_full
bash scripts/ablation/test_ablation.sh
bash scripts/ablation/eval_ablation.sh
```

Variants: `abl_full`, `abl_no_gan`, `abl_no_icc`, `abl_no_idt`,
`abl_supervised_only`, `abl_mse_lambda_1`, `abl_mse_lambda_10`.

## Downstream tasks

Quality grading, vessel segmentation and OD/OC + fovea detection are evaluated with
third-party toolkits. They are not redistributed here; see [downstream/README.md](downstream/README.md).

## Results

See [docs/results.md](docs/results.md) for the numbers reported in the paper and the
reproduction status of each table.

## Citation

```bibtex
@article{semiet2026,
  title   = {SemiET: Semi-supervised fundus enhancement with efficient Transformers},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Article reference BSPC 111659},
  author  = {Qiu, Jiaming and Xu, Feng and Ren, Yiwei and Cheng, Pujin and Tang, Xiaoying}
}
```

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgement

This code builds on the MICCAI 2021 **I-SECRET** repository, Restormer, CUT, and
EyeQ_Enhancement. We thank the authors for releasing their code.
