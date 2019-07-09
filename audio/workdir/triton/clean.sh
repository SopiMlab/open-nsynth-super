#!/bin/bash
N="$1"
PROJECT_DIR="$2"

cd "$PROJECT_DIR/audio/workdir"

if [ ! -d audio_output/raw_wav ]; then
       	mkdir audio_output/raw_wav
       	mv audio_output/batch* audio_output/raw_wav
fi

for ((i=0; i < N; i++)); do
       	python 05_clean_files.py "$i" &
	pids[${i}]=$!
done

for pid in ${pids[*]}; do
	wait $pid
done
