# Third-party notices

This repository contains code adapted from the following open-source projects.
Please refer to each upstream repository for the full license text.

| Component | Where | Upstream | License (check upstream) |
|---|---|---|---|
| I-SECRET (MICCAI 2021) | base repository | https://github.com/MVP-D77/ISECRET | MIT |
| Restormer | `isecret/model/restormer_network.py` | https://github.com/swz30/Restormer | MIT |
| CUT / patch-wise contrastive loss | `isecret/loss/icc_loss.py` | https://github.com/taesungp/contrastive-unpaired-translation | MIT |
| EyeQ_Enhancement (degradation) | `tools/degrade_eyeq.py` | https://github.com/HzFu/EyeQ_Enhancement | see upstream |
| SlowFast (distributed helpers) | `isecret/utils/` | https://github.com/facebookresearch/SlowFast | Apache-2.0 |

The downstream evaluation in the paper uses third-party toolkits
(MCF-Net, IterNet-style vessel segmentation, and the fovea/OD-OC detector).
They are **not** redistributed here; see `downstream/README.md`.
