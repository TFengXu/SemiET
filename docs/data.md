# Data preparation

## EyeQ

Download EyeQ from the official repository (https://github.com/HzFu/EyeQ) and split it
into train/val/test following the EyePACS challenge. The split lists used in this project
are `train.csv`, `val.csv`, and `test.csv` (columns: `name,label`).

Arrange the high-quality crops as:

```
data/eyeq/
├── train/{crop_good,crop_usable,mask_good}
├── val/{crop_good,crop_usable,mask_good}
└── test/{crop_good,crop_usable,mask_good}
```

Then synthesize the degraded pairs:

```bash
python tools/degrade_eyeq.py \
    --test_dir data/eyeq/train/crop_good \
    --gt_dir   data/eyeq/train/crop_good \
    --mask_dir data/eyeq/train/mask_good \
    --output_dir data/eyeq/train/degrade_good
```

Repeat for `val` and `test`. The degradation model follows CofeNet / EyeQ_Enhancement
(light transmission disturbance, defocus blur, retinal artifacts).

## External test sets

Out-of-domain evaluation uses **REFUGE**, **DRIVE**, **CHASE-DB1**, **GAMMA** and **FIVES**.
Download them from their official sources, apply the same degradation script, and pass the
resulting folder to `scripts/test_semiet.sh`. We do not redistribute any dataset.

## Directory expected by the code

`--data_root_dir` should point to the folder that contains `eyeq/`, i.e. `data`.
`--dataset eyeq` then resolves to `data/eyeq`.
