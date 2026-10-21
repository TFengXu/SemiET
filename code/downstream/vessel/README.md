# Vessel segmentation

Upstream: IterNet-style vessel segmentation (https://github.com/conscienceli/IterNet).

Protocol used in the paper: degrade the DRIVE and CHASE-DB1 images with
`tools/degrade_eyeq.py`, enhance them with SemiET, then run vessel segmentation on the
enhanced images and compute Dice against the vessel masks.

The upstream `test_vessel.py` is the entry point; set its test directory to the enhanced
images and its checkpoint to the IterNet model.
