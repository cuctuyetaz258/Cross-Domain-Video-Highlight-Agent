# Kaggle V2 Setup

`run_v2.py` is a private GPU Kaggle Script. Its first pushed version only
validates the private `video-highlight-v2-inputs` Dataset. This prevents an
accidental random-initialized five-fold run.

To run the scientific V2 experiment, first create a second private Dataset
containing the schema-1.1 TVSum+SumMe cache and the resulting V2 pretrained
checkpoint. Add that Dataset as `nguyentrann0703/video-highlight-v2-pretrain`,
change `RUN_MODE` to `five_fold`, and push a new kernel version.

For reliable unattended runs, use `RUN_MODE = "fold"` and change `FOLD_INDEX`
from 0 through 4, saving one Kaggle version per fold. Each epoch writes both a
best checkpoint and a `*_last.pt` recovery checkpoint, while a successful fold
publishes `v2_tcn_ltr_foldN_artifacts.zip` in Kaggle output. Kaggle does not
guarantee `/kaggle/working` files survive a timeout or failed version, so do
not use one long five-fold version when artifacts must be retained.
