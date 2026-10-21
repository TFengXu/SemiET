# Quality grading (MCF-Net)

Upstream: EyeQ / MCF-Net (https://github.com/HzFu/EyeQ).

Input: enhanced EyeQ `usable` images, e.g. `data/eyeq/test/degrade_semiet_enhance`.
Output: counts of predicted Good / Usable / Reject, plus FIQA / WFQA.

The exact inference entry point used in the paper is `Main_EyeQuality.py` from the
upstream MCF-Net code. Point its `test_dir` to the enhanced folder and run it with the
MCF-Net checkpoint.

Note: the original script contains absolute paths from our cluster; replace them with
your local paths before running.
