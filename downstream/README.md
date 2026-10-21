# Downstream evaluation

The paper evaluates enhancement quality with three downstream tasks. The corresponding
toolkits are third-party projects and are **not** redistributed in this repository.
Each sub-folder documents the upstream project, the expected input layout and the command
we used.

- `mcf_net/` - EyeQ quality grading (MCF-Net), used for the non-reference table.
- `vessel/` - retinal vessel segmentation (IterNet-style), used for DRIVE / CHASE-DB1 Dice.
- `fovea/` - OD/OC segmentation and fovea localization, used for REFUGE / GAMMA.

All of them take the enhanced images produced by `scripts/test_semiet.sh`.
