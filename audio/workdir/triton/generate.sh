#!/bin/bash
COUNT=$1
START=$2
PROJECT_DIR=$3

if [ ! -n "$COUNT" ] || [ ! -n "$START" ] || [ ! -n "$PROJECT_DIR" ]; then
    echo "usage: generate.sh count start project_dir"
    exit 1
fi

for ((i=START; i < START+COUNT; i++)); do
	nsynth_generate \
		--checkpoint_path="$PROJECT_DIR/magenta/magenta/models/nsynth/wavenet-ckpt/model.ckpt-200000" \
		--source_path="$PROJECT_DIR/audio/workdir/embeddings_batched/batch$i" \
		--save_path="$PROJECT_DIR/audio/workdir/audio_output/batch$i" \
		--batch_size=2048 \
		--alsologtostderr \
		--gpu_number="$((i-START))" \
                --samples_per_save=300000 &
	pids[${i}]=$!
done

for pid in ${pids[*]}; do
	wait $pid
done
