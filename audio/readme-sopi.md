# Open NSynth Super audio generation

This guide is intended to supplement [Google's](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio).

The audio generation pipeline has seemingly only been tested on Linux. It can probably be made to work on other platforms, but there is e.g. some path manipulation code that assumes Unix paths and will break on Windows.

Our reference computer's hardware specifications are:

- CPU: Intel Core i7-6700HQ (4 cores)
- RAM: 16 GB
- GPU: NVIDIA GeForce GTX 1080 Ti (connected via external Thunderbolt enclosure)

Ideally, your system will have even more of everything; multiple GPUs are particularly helpful. Note that CUDA, which is required, only works with NVIDIA GPUs.

Software versions tested:

- Ubuntu 18.04 LTS
- NVIDIA driver 418.67
- CUDA 10.1
- Python 2.7.16
- tensorflow-gpu 1.14.0
- magenta-gpu 1.1.2

## Install Linux

We will use the popular [Ubuntu](https://ubuntu.com/) distribution.

As of July 2019, it is not possible to access GPUs through a Linux virtual machine or Docker container running on a Windows host, so the only feasible option is a native Linux installation.

Follow the [Ubuntu installation guide](https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop).

## Install NVIDIA drivers

By default, Ubuntu uses the open-source nouveau driver for NVIDIA graphics. We need NVIDIA's driver instead.

Add the graphics drivers package archive to APT:

```
sudo add-apt-repository ppa:graphics-drivers-ppa
sudo apt update
```

At this point, you may want to do a full upgrade of all installed packages to make sure everything is up to date:

```
sudo apt upgrade
```

Now, open Software & Updates and go to the Additional Drivers tab. Select the latest NVIDIA driver from the list and click Apply Changes. A reboot may be required.

## Install required packages

We need Git as well as some header files:

```
sudo apt install git
sudo apt install libasound2-dev
sudo apt install libjack-jackd2-dev
```

## Install CUDA

The [CUDA](https://developer.nvidia.com/cuda-zone) toolkit is required to run the GPU-accelerated version of TensorFlow.

Follow [NVIDIA's CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), making sure to pay attention to the Ubuntu-specific parts.

## Install Miniconda

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a minimal variant of the Anaconda Python distribution.

Follow the [Miniconda installation guide for Linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

## Set up a conda environment

Create a conda environment with Python 2.7 and tensorflow-gpu (we'll name it "open-nsynth-super"):

```
conda create -n open-nsynth-super python=2.7 tensorflow-gpu=1.13
```

Activate the environment:

```
conda activate open-nsynth-super
```

Install magenta-gpu:

```
pip install magenta-gpu
```

## Check GPU access

TensorFlow should now be able to access your GPU. To test this, paste the following code into a new file called `listgpus.py`:

```
from tensorflow.python.client import device_lib

for device in device_lib.list_local_devices():
	if device.device_type == "GPU":
		print(device)
		print()
```

and run it:

```
python listgpus.py
```

If you can see your GPU in the printed list, you're good to go!

TODO: provide this in our own repository

## Download code

Clone the [open-nsynth-super repository](https://github.com/googlecreativelab/open-nsynth-super):

```
git clone https://github.com/googlecreativelab/open-nsynth-super.git
```

Going forward, we'll assume the cloned repository is in our home directory: `/home/sopi/open-nsynth-super` — make sure to adjust paths according to your system.

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

Follow the [Preparing to run the pipeline](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio#preparing-to-run-the-pipeline) section of Google's guide to prepare audio files and edit `settings.json`.

The `magenta_dir` setting should be set to the magenta directory you created. In my case this is `/home/sopi/open-nsynth-super/magenta`.

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

## Generate

You should now be ready to run all the scripts! Follow the [Running the pipeline](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio#running-the-pipeline) section of Google's guide.

`samples_per_save=300000`