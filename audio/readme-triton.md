# Open NSynth Super audio generation on Triton

The Aalto Science-IT [Triton](https://scicomp.aalto.fi/triton/) computing cluster includes nodes with multiple fast GPUs.

Running jobs on Triton is a bit different from using a regular Linux system, so the [Open NSynth Super audio generation](readme-sopi.md) guide does not apply directly here, but it may still be helpful to reference.

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
conda create -p "$WRKDIR/conda/open-nsynth-super" python=2.7 tensorflow-gpu=1.14
module unload teflon
```

Activate the environment:

```
conda activate "$WRKDIR/conda/open-nsynth-super"
```

## Download code

Enter your work directory:

```
cd "$WRKDIR"
```

Clone our [open-nsynth-super repository](https://github.com/SopiMlab/open-nsynth-super.git):

```
git clone https://github.com/SopiMlab/open-nsynth-super.git
```

## Install magenta-gpu

Enter the open-nsynth-super repository directory:

```
cd open-nsynth-super
```

Clone our magenta repository and enter the resulting directory:

```
git clone https://github.com/SopiMlab/magenta.git
cd magenta
```

Building the magenta-gpu package requires Python 3, so create and activate another conda environment for this:

```
module load teflon
conda create -p "$WRKDIR/conda/magenta-build" python=3.7 tensorflow-gpu=1.14
module unload teflon
conda activate "$WRKDIR/conda/magenta-build"
```

Build the package:

```
python setup.py bdist_wheel --universal --gpu
```

This should create a file in the `dist` directory called something like `magenta_gpu-1.1.7-py2.py3-none-any.whl`.

Switch back to the Python 2 environment and install the package:

```
conda deactivate
# run this line if you're not in open-nsynth-super at this point:
conda activate "$WRKDIR/conda/open-nsynth-super"

module load teflon
pip install dist/magenta_gpu-1.1.7-py2.py3-none-any.whl
module unload teflon
```

If you want, you can now remove the Python 3 environment:

```
conda env remove -p "$WRKDIR/conda/magenta-build"
```

## Download checkpoint

Download the NSynth WaveNet checkpoint into the appropriate directory and extract it:

```
cd magenta/models/nsynth
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

The `gpus` setting should reflect the total number of GPUs on the node(s) you intend to use for generation. E.g. if you use two nodes with 4 GPUs each, specify 8.

See the [Triton cluster overview](https://scicomp.aalto.fi/triton/overview.html) for a list of available nodes.

Once done, enter the audio work directory:

```
cd audio/workdir
```

## Generate

The `triton` directory contains Slurm and shell scripts for use on Triton. These invoke the NSynth scripts as described in the [Running the pipeline](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio#running-the-pipeline) section of Google's guide.

Before running the scripts, you may want to read the [Tuning](#section) section below as well as the notes in our [main generation guide](readme-sopi.md#generate)!

### Monitoring status

There are a few useful commands available for monitoring jobs. On the login node, you can run:

- `slurm watch queue`, to view the status of your job queue.
- `tail -F slurm-<JOBID>.out`, to see real-time output from the job. (Each job writes an output file in the directory where `sbatch` was run)

Once the job has started, `slurm watch queue` will show the name of the node where it's running. You can connect to this node over SSH (through the Triton login node) and monitor resource usage with:

- `htop`, to see CPU and memory use
- `watch -n 1 nvidia-smi`, to see GPU use

### Deactivate the conda environment!

Having a conda environment active when submitting jobs can mess up Python paths, so make sure to deactivate first.

```
conda deactivate
```

### Compute and batch embeddings

The `prepare` script handles steps 1-3 (computing and batching embeddings):

```
sbatch triton/prepare.slrm
```

This job should take no longer than a few minutes to complete. Afterwards, you should have a subdirectory in `embeddings_batched` for each GPU, named `batch0`, `batch1` etc.

### Generate audio

To generate, you will need to submit a job for each node you intend to use, with a command like:

```
sbatch triton/generate.slrm <BATCH_COUNT> <BATCH_START>
```

where `BATCH_COUNT` specifies the number of batches to process on the node, typically equal to the number of GPUs on the node, and `BATCH_START` specifies the number of the node to start from.

As an example, say we want to run on two nodes, with 4 GPUs each. The commands would be:

```
sbatch triton/generate.slrm 4 0
sbatch triton/generate.slrm 4 4
```

The first node will process batches 0,1,2,3 while the second one will do batches 4,5,6,7.

Generation will typically take several days.

### Clean files

Once generation has finished, the audio files need to be cleaned up. Run:

```
sbatch triton/clean.slrm
```

### Build pads

Finally, to build the `.bin` file(s), you can again activate the conda environment and run `06_build_pads.py` directly:

```
conda activate "$WRKDIR/conda/open-nsynth-super"
python 06_build_pads.py
```

## Transfer pad files and deploy

Use one of the methods described in the [Triton data storage tutorial](https://scicomp.aalto.fi/triton/tut/storage.html#accessing-and-transferring-files-remotely) to transfer the resulting `.bin` file(s) from the `pads_output` directory to your local computer. Then follow the deploy section of the [main guide](readme-sopi.md).

## Tuning

System requirements for Triton nodes are specified in the Slurm scripts. Our first audio generation test completes in about 4 days, using:

- 4 instruments (1 per corner)
- 4 notes per instrument
- 1 node with 4 Tesla V100 GPUs

Triton's maximum job duration is 5 days, so larger input sample sets will likely exceed the limit. It should be fairly simple to split the task across multiple nodes to speed it up.

This however seems much slower than what Google's guide implies. During generation, resource usage is as follows:

- CPU load: ~60% on each of the 10 CPUs
- Memory: ~50GB out of 64GB reserved
- GPU load: ~75% on each of the 4 GPUs
- GPU memory: ~1.5GB out of 32GB

More investigation is needed to determine what exactly the limiting factor is here and whether we can improve on the performance.