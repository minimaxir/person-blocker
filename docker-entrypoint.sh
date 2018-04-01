#!/bin/sh
set -e
xvfb-run python -W ignore /app/person_blocker.py -m /app/mask_rcnn_coco.h5 "$@"
