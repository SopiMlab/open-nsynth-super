# Open NSynth Super audio generation on Triton

The Aalto Science-IT [Triton](https://scicomp.aalto.fi/triton/) computing cluster includes nodes with multiple fast GPUs.

Running jobs on Triton is a bit different from using a regular Linux system, so the [Open NSynth Super audio generation](generating-audio.md) guide does not apply directly here, but it may still be helpful to reference.

To use Triton, you need to [request access](https://scicomp.aalto.fi/triton/accounts.html). It also requires some familiarity with the Linux command line and the Slurm job scheduling system. They have [tutorials](https://scicomp.aalto.fi/triton/tut/intro.html). You'll at least want to go through these:

- Connecting to Triton
- Software Modules
- Applications: Python
- Data storage
- Interactive jobs
- Serial jobs
- GPU computing

## Set up a conda environment

Log in to Triton using SSH. There are a variety of ready-to-use Anaconda modules available. To list Anaconda2 versions, run:

```
module spider anaconda2
```

It's not entirely clear which available version is the most appropriate, or to what extent it makes a difference since we will be installing our own packages anyway. I have picked `anaconda2/5.1.0-gpu` somewhat arbitrarily.

To load the module, run:

```
module load anaconda2/5.1.0-gpu
```

Create the conda environment on the scratch file system (refer to the [Python tutorial](https://scicomp.aalto.fi/triton/apps/python.html)) and activate it:

```
mkdir "$WRKDIR/conda"
module load teflon
conda create --prefix "$WRKDIR/conda/open-nsynth-super" python=2.7 tensorflow-gpu
module unload teflon
```

Activate the environment and install magenta-gpu:

```
source activate "$WRKDIR/conda/open-nsynth-super"
module load teflon
pip install magenta-gpu
module unload teflon
```

## Download code

Enter your work directory:

```
cd "$WRKDIR"
```

Clone the [open-nsynth-super repository](https://github.com/googlecreativelab/open-nsynth-super):

```
git clone https://github.com/googlecreativelab/open-nsynth-super.git
```

Enter the repository directory:

```
cd open-nsynth-super
```

Google says to clone the Magenta repository here, but in fact we only need a small part of its directory structure. Create it manually:

```
mkdir -p magenta/magenta/models/nsynth
```

Download and extract the NSynth WaveNet checkpoint:

```
cd magenta/magenta/models/nsynth
wget http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar
tar xf wavenet-ckpt.tar
```

You may want to remove the now redundant archive file:

```
rm wavenet-ckpt.tar
```

Go back to the `open-nsynth-super` root:

```
cd ../../../..
```

## Prepare for generation

Follow the [Preparing to run the pipeline](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio#preparing-to-run-the-pipeline) section of Google's guide to prepare audio files and edit `settings.json`. The Triton storage tutorial explains how to transfer the audio files over.

The `magenta_dir` setting should be set to the magenta directory you created. In my case this is `/scratch/work/kastemm1/open-nsynth-super/magenta`.

The `gpus` setting should reflect the number of GPUs available on the node you intend to use. See the [Triton cluster overview](https://scicomp.aalto.fi/triton/overview.html) for a list of available nodes. So far, I've only tested on the gpu[28-37] nodes, which have 4 GPUs.

Enter the audio work directory:

```
cd audio/workdir
```

## Fix Google's code

As of July 2019, there appears to be a [bug](https://github.com/googlecreativelab/open-nsynth-super/issues/77) in the generation code. Edit the file `02_compute_new_embeddings.py` and find the line:

```
interpolated = np.reshape(interpolated, (1,) + interpolated.shape)
```

Remove this line or comment it out by adding a `#` at the start.

TODO: provide our own modified fork of the repo

### Progress reporting (optional but useful)

The nsynth_generate script is not very helpful about reporting progress. This can be improved by making a small modification to Magenta's `fastgen.py` file, found deep in the conda environment's libraries directory; in our case at `/home/sopi/miniconda3/envs/open-nsynth-super/lib/python2.7/site-packages/magenta/models/nsynth/wavenet/fastgen.py`.

In the function `synthesize()`, find the line:

```
tf.logging.info("Sample: %d" % sample_i)
```

and replace it with:

```
tf.logging.info("Sample: {}/{} ({:.1f}%)".format(sample_i, total_length, float(sample_i)/total_length*100))
```

making sure to keep the indentation intact. The script will now print completion percentages when generating samples.

TODO: provide a magenta-gpu package that includes this?

## Prepare Slurm scripts

Create the following three files in the audio work directory:

**prepare.slrm**

```
#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

PROJECT_DIR="$WRKDIR/open-nsynth-super"
CONDA_ENV="$WRKDIR/conda/open-nsynth-super"

cd "$PROJECT_DIR/audio/workdir"
module load anaconda2/5.1.0-gpu
source activate "$CONDA_ENV"

srun python 01_compute_input_embeddings.py
srun python 02_compute_new_embeddings.py
srun python 03_batch_embeddings.py
```

**generate.sh**

```
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
```

**generate.slrm**

```
#!/bin/bash
#SBATCH --time=5-00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:4

PROJECT_DIR="$WRKDIR/open-nsynth-super"
CONDA_ENV="$WRKDIR/conda/open-nsynth-super"

cd "$PROJECT_DIR/audio/workdir"
module load anaconda2/5.1.0-gpu
source activate "$CONDA_ENV"

srun ./generate.sh 4 "$PROJECT_DIR"
```

## Run scripts

First, generate embeddings:

```
sbatch prepare.slrm
```

Once this job is complete, start audio generation:

```
sbatch generate.slrm
```

`prepare.slrm` should complete in a few minutes, `generate.slrm` will take much longer.

There are a few useful commands for monitoring the job. On the login node, you can run:

- `slurm watch queue`, to view the status of your job queue.
- `tail -F slurm-<JOBID>.out`, to see real-time output from the job. (Each job generates an output file in the directory where `sbatch` was run)

Once the job has started, `slurm watch queue` will show the name of the node where it's running. You can connect to this node over SSH (through the Triton login node) and monitor resource usage with:

- `htop`, to see CPU and memory use
- `watch -n 1 nvidia-smi`, to see GPU use

TODO: next step

## Results

Currently it looks like my first attempt at audio generation will complete in a bit less than 5 days, which is the maximum job duration on Triton. This is using only 1 instrument per corner and 4 samples per instrument. Larger input data will likely exceed the limit.

Google's guide suggests that this is actually significantly slower than expected. Looking at resource usage currently, I see:

- CPU load: ~60% on each of the 10 CPUs
- Memory: ~50GB out of 64GB reserved
- GPU load: ~75% on each of the 4 GPUs
- GPU memory: ~1.5GB out of 32GB

It's not clear to me what the bottleneck might be here. Some ideas for improvement:

- Saving the intermediate WAV files takes quite a bit of time, consider doing it less frequently (`fastgen.generate()` has a `samples_per_save` parameter for this)
- Also, use `/tmp` for file storage during processing (faster than scratch), and only move things to scratch at the end
- I tried tweaking `batch_size` with no apparent difference, but this needs more investigation