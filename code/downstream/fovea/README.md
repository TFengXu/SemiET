# OD/OC segmentation and fovea localization

The paper uses a multi-task detector for optic disc/cup segmentation and fovea center
localization on the degraded and enhanced REFUGE and GAMMA images.

The evaluation notebooks (`calculate_refuge.ipynb`, `calculate_gamma.ipynb`) are part of
the internal codebase and are not redistributed. The protocol is:

1. degrade REFUGE / GAMMA with `tools/degrade_eyeq.py`;
2. enhance with `scripts/test_semiet.sh`;
3. run the detector and compute ODSC / OCSC / FMAE.

Please contact the authors for the exact detector version used in the paper.
