# Open NSynth Super audio generation

This guide is intended to supplement [Google's](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio) guide.

The audio generation pipeline seems to have been tested only on Linux. It can probably be made to work on other platforms, but there is e.g. some path manipulation code that assumes Unix style paths and will break on Windows.

Our reference computer's hardware specifications are:

- CPU: Intel Core i7-6700HQ (4 cores)
- RAM: 16 GB
- GPU: NVIDIA GeForce GTX 1080 Ti (connected via external Thunderbolt enclosure)

Ideally, your system will have more of everything; multiple GPUs and more RAM are particularly helpful. Note that CUDA, which is required, only works with NVIDIA GPUs.

Software versions tested:

- Ubuntu 18.04 LTS
- NVIDIA driver 418.67
- CUDA 10.1
- Python 2.7.16
- tensorflow-gpu 1.13.0
- magenta-gpu based on 1.1.2

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

The audio generation pipeline requires a few packages to work. Install them using APT:

```
sudo apt install git libasound2-dev libjack-jackd2-dev ffmpeg sox lame libsox-fmt-mp3
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

## Download code

Clone this repository:

```
git clone TODO
```

Going forward, we'll assume the cloned repository is in our home directory: `/home/sopi/open-nsynth-super` — make sure to adjust paths according to your system.

## Install magenta-gpu

Enter the open-nsynth-super repository directory:

```
cd open-nsynth-super
```

Clone our magenta repository and enter the resulting directory:

```
git clone TODO
cd magenta
```

Building the magenta-gpu package requires Python 3, so create and activate another conda environment for this:

```
conda create magenta-build python=3.7 tensorflow-gpu=1.13
conda activate magenta-build
```

Build the package:

```
python setup.py bdist_wheel --universal --gpu
```

This should create a file in the `dist` directory called something like `magenta_gpu-1.1.2-py2.py3-none-any.whl`.

Switch back to the Python 2 environment and install the package:

```
conda activate open-nsynth-super
pip install dist/magenta_gpu-1.1.2-py2.py3-none-any.whl
```

If you want, you can now remove the Python 3 environment:

```
conda env remove -n magenta-build
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

## Check GPU access

TensorFlow should now be able to access your GPU. To test this, enter the audio work directory and run the `listgpus.py` script:

```
cd audio/workdir
python listgpus.py
```

If you can see your GPU(s) in the printed list, you're good to go!

## Prepare for generation

Follow the [Preparing to run the pipeline](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio#preparing-to-run-the-pipeline) section of Google's guide to prepare audio files and edit `settings.json`.

The `magenta_dir` setting should be set to your magenta directory. In my case this is `/home/sopi/open-nsynth-super/magenta`.

## Generate

You should now be ready to run all the scripts! Follow the [Running the pipeline](https://github.com/googlecreativelab/open-nsynth-super/tree/master/audio#running-the-pipeline) section of Google's guide, but have a look also at the notes below.

### 1. Compute input embeddings

If you get an out of memory error, you can edit the `--batch_size` parameter on the last line of the script to use a smaller value than the default 64. However, any system where this is an issue will probably be very slow at generation.

### 2. Compute interpolated embeddings

Our modified version of this script fixes a [bug](https://github.com/googlecreativelab/open-nsynth-super/issues/77) where this step generates incorrectly shaped data.

### 3. Batch embeddings for processing

Nothing special here, just moving files around.

### 4. Generate audio

You can run this manually for each GPU, as suggested in Google's guide, or use the `generate.sh` script in the `triton` directory, passing the number of GPUs and the path to the `open-nsynth-super` directory as arguments, e.g.:

```
./triton/generate.sh 4 /home/sopi/open-nsynth-super
```

This would start 4 concurrent instances of `nsynth_generate`, using the GPUs with index 0, 1, 2 and 3 respectively.

The `--batch_size` parameter has a big impact on performance. Our modified `nsynth_generate` prints out some useful progress information while running:

```
0.4% - Batch: 1/5 - Sample: 609001/32768000
```

For optimal performance, you'll want to minimize the total number of batches (5 in the example above) by increasing the batch size. Since the exact numbers will vary depending on the number of audio files, number of GPUs etc., and larger batch sizes also require more memory, you may need to experiment a bit.

Another performance consideration is how often to save intermediate results to disk, as the saving can take a significant amount of time, during which generation is stalled. By default, `nsynth_generate` will save after every 10000 samples generated, which is quite frequent and can waste a lot of time.

With our modified `nsynth_generate`, the save interval can be adjusted using the `--samples_per_save` parameter. I've found a value of 300000 to be reasonable.

### 5. Clean files

This is where the ffmpeg, sox, lame and libsox-fmt-mp3 packages (installed earlier) are needed.

It's not clear why this step involves a WAV-to-MP3-to-WAV conversion. Maybe they're using MP3 as a low-pass filter to remove high frequency noise?

### 6. Build pads

Not sure whether it's my mistake, but this step gave me a file with the instruments in a different order than I expected, i.e. `bird1_bird2_bird4_bird3.bin` instead of `bird1_bird2_bird3_bird4.bin`. It's worth checking the output, because the order matters for the next step.

### 7. Deploy to the device

Google's guide technically explains how to do this, but the info is scattered across multiple documents. Here's a simpler version!

Upon inserting your SD card into the computer, you'll see multiple partitions. On the largest partition, browse into `/home/pi` and create a directory for your sounds, e.g. `birds_audio` in my case. Copy the `.bin` file(s) from the previous step into this directory.

Next, browse into `/home/pi/opt/of/apps/open-nsynth/open-nsynth/bin/data`. This folder contains the `settings.json` file to be edited, which you may want to back up first.

Set `dataDirectory` to the path of the audio directory you created.

In `pitches`, specify all the note numbers you used for your input audio files.

Set your desired sample loop points with `loopStart` and `loopEnd` (specified in seconds) — or set `looping` to `false` to disable looping.

In `corners`, enter your instruments for each corner of the touch pad. **Note that the order here needs to match the order in which the instruments appear in the name of your `.bin` file(s)!** `name` is the audio file name (without extension), `display` is the name shown on the synth while selecting instruments and `abbr` is the short name shown in the corners of the display when playing.

Example:

```
{
    "debug": false,
    "nsynth": {
        "dataDirectory": "/home/pi/birds_audio",
        "resolution": 9,
        "pitches": [24, 27, 30, 33],
        "length": 60000,
        "sampleRate": 16000,
        "looping": true,
        "loopStart": 0.0,
        "loopEnd": 4.0
    },
    "corners": [
        {
            "instruments": [
                {"name": "bird1", "display": "BIRD1", "abbr": "B1"}
            ]
        },
        {
            "instruments": [
                {"name": "bird2", "display": "BIRD2", "abbr": "B2"}
            ]
        },
        {
            "instruments": [
                {"name": "bird4", "display": "BIRD4", "abbr": "B4"}
            ]
        },
        {
            "instruments": [
                {"name": "bird3", "display": "BIRD3", "abbr": "B3"}
            ]
        }
    ],
    "audio": {
        "sampleRate": 48000,
        "bufferSize": 128
    },
    "midi": {
        "device": "/dev/ttyAMA0",
        "channel": 1
    },
    "osc": {
        "in": {
            "portNumber": 8000
        }
    },
    "patchFile": "/media/data/patches.json"
}
```

Finally, save the file, eject the SD card and insert it back into the Open NSynth Super. You should now be able to play it with your sounds!