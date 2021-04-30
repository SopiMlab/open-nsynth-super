# Open NSynth Super — generating on Paniikki

## Log in to Paniikki

Follow our [Using Aalto computers](https://github.com/SopiMlab/DeepLearningWithAudio/blob/master/using-aalto-computers.md) guide to log in to Paniikki and start a screen.

## Clone repositories

```
git clone https://github.com/SopiMlab/open-nsynth-super.git
cd open-nsynth-super
git clone https://github.com/SopiMlab/magenta.git
```

## Load modules

```
module load anaconda3
```

## Create Conda environment

```
conda env create -n open-nsynth-super -f open-nsynth-super.yml
source activate open-nsynth-super
```

## Install Magenta

Enter the Magenta folder:

```
cd magenta
```

Check the full path of this folder and write it down (it will be needed later):

```
pwd
```

Check out a Python 2 compatible version:

```
git checkout 172a9cb
```

This will warn you about "detached HEAD", but that's fine.

Remove `python-rtmidi` from Magenta's dependencies:

```
sed -e '/python-rtmidi/ s/^#*/#/' -i setup.py
```

Building the `magenta-gpu` package requires Python 3, so create and activate another conda environment for this:```conda create -n magenta-build python=3.7 tensorflow-gpu=1.13```

Activate the `magenta-build` environment:

```source activate magenta-build```Build the package:```python setup.py bdist_wheel --universal --gpu```This should create a file in the `dist` directory called something like `magenta_gpu-1.1.7-py2.py3-none-any.whl`.Switch back to the Python 2 environment and install the package:```source activate open-nsynth-superpip install dist/magenta_gpu-1.1.7-py2.py3-none-any.whl```

If you want, you can now remove the `magenta-build` environment:```conda env remove -n magenta-build```

## Download checkpoint

Enter the `open-nsynth-super/magenta/magenta/models/nsynth` directory:

```
cd magenta/models/nsynth
```

Download the NSynth WaveNet checkpoint:

```
wget http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar
```

Extract it:

```
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

The `magenta_dir` setting should be set to your magenta directory. (The path you got from `pwd` previously)

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

Not sure whether it's my mistake, but this step gave me a file with the instruments in a different order than I expected, i.e. `bird1_bird2_bird4_bird3.bin` instead of `bird1_bird2_bird3_bird4.bin`.
