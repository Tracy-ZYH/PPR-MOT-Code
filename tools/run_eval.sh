#!/bin/bash

# Configuration of paths
DATA_ROOT="results/TrackEval_converted"
TRACKEVAL_PATH="TrackEval-master" # Adjust this to your TrackEval directory

# Check if TrackEval exists
if [ ! -d "$TRACKEVAL_PATH" ]; then
    echo "[ERROR] TrackEval directory not found at: $TRACKEVAL_PATH"
    exit 1
fi

echo "[INFO] Starting PPR-MOT evaluation..."

# Execute MOTChallenge evaluation
python3 "$TRACKEVAL_PATH/scripts/run_mot_challenge.py" \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL predict \
    --METRICS HOTA \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 4 \
    --GT_FOLDER "$DATA_ROOT/gt/train" \
    --TRACKERS_FOLDER "$DATA_ROOT/trackers" \
    --SEQMAP_FILE "$DATA_ROOT/seqmap.txt" \
    --SKIP_SPLIT_FOL True \
    --PLOT_CURVES False

echo "[SUCCESS] Evaluation finished."