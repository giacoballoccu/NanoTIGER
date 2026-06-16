#!/usr/bin/env bash
# run.sh -- the whole nanoTIGER pipeline, one command.
#
#   bash run.sh                      # full pipeline with config.py defaults
#   bash run.sh --category All_Beauty   # override the dataset category
#
# Stops at the first error. Each stage prints "Next: ..." so you can also run
# them one at a time (see the README).
set -euo pipefail

CATEGORY_ARG="${@:-}"

echo "==> [1/5] prepare_data.py"
python prepare_data.py ${CATEGORY_ARG}
echo "==> [2/5] embed_items.py"
python embed_items.py
echo "==> [3/5] rqvae.py"
python rqvae.py
echo "==> [4/5] train.py"
python train.py
echo "==> [5/5] eval.py"
python eval.py
echo "==> done. Try: python show_neighbors.py"
