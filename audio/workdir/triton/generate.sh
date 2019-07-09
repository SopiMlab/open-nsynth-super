#!/bin/bash
N="$1"
PROJECT_DIR="$2"

for ((i=0; i < N; i++)); do
	nsynth_generate \
		--checkpoint_path="$PROJECT_DIR/magenta/magenta/models/nsynth/wavenet-ckpt/model.ckpt-200000" \
		--source_path="$PROJECT_DIR/audio/workdir/embeddings_batched/batch$i" \
		--save_path="$PROJECT_DIR/audio/workdir/audio_output/batch$i" \
		--batch_size=2048 \
		--alsologtostderr \
		--gpu_number="$i" &
	pids[${i}]=$!
done

for pid in ${pids[*]}; do
	wait $pid
done
