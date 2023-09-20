# Genomic Prior-Data Fitted Networks

This repo provides a codebase for building and deploying Genomic Prior-Data Fitted Networks (GPFNs).

If you use this code, please cite [todo].

A few functions in this code were lifted from [the official implementation](https://github.com/automl/TransformersCanDoBayesianInference) of Prior-Data Fitted Networks by Müller et al. Where this is the case, a comment should accompany.

## Table of Contents

- [Running Models Locally With Docker](#running-models-locally-with-docker)
  * [Installing Docker](#installing-docker)
  * [Available GPFNs](#available-gpfns)
  * [Step 1: Parse your training and target data.](#step-1--parse-your-training-and-target-data)
  * [Step 2: Do Inference](#step-2--do-inference)
- [Fitting Models in Python](#fitting-models-in-python)
  * [Installation](#installation)
  * [Fitting New Models Locally](#fitting-new-models-locally)
  * [Fitting New Models on HPC](#fitting-new-models-on-hpc)
    + [Option 1: Generating a Cached Dataset on a Single Node](#option-1--generating-a-cached-dataset-on-a-single-node)
    + [Option 2: Generating a Cached Dataset on Multiple Nodes with MPI](#option-2--generating-a-cached-dataset-on-multiple-nodes-with-mpi)
    + [Option 3: Generating Data with MPI and Fitting in Parallel](#option-3--generating-data-with-mpi-and-fitting-in-parallel)
  * [Making New Priors](#making-new-priors)
- [License](#license)

## Running Models Locally With Docker

***⚠️ If you just want to use GPFNs with your own data, this is the easiest option. ⚠️***

There is a [Docker image](https://hub.docker.com/repository/docker/jubbens) provided for those who want to use trained GPFNs. Because of GitHub's git-lfs bandwidth limits, this is currently the only way to download the trained models.

### Installing Docker

First, you should [install docker](https://docs.docker.com/) according to the directions for your own system. If you have a CUDA-capable GPU that you want to use, you will also need to [install the nvidia container runtime](https://docs.nvidia.com/ai-enterprise/deployment-guide-vmware/0.1.0/docker.html). If you don't, that's fine! Your CPU will be used instead. The first time you call a `docker run` command, the image will be downloaded. This will take a while, but it will only happen once. 

### Available GPFNs

Below are the currently offered fit models. All of them are included in the docker image.

| Name     | Path                 | Compatible Population Types                              | Parameters | File Size |
|----------|----------------------|----------------------------------------------------------|------------|-----------|
| Pika     | `deploy/pika.pt`     | Nested Association Mapping (between-families prediction) | 311M       | 1.16 GB   |
| Mongoose | `deploy/mongoose.pt` | Unstructured (diverse)                                   | 311M       | 1.16 GB   |
| Wombat   | `deploy/wombat.pt`   | da Silva et al. [^1]                                     | 311M       | 1.16 GB   |

[^1]: Implements the genomic selection scheme proposed for Soybean in [(da Silva et al. 2021)](https://doi.org/10.3389/fgene.2021.637133).

### Step 1: Parse your training and target data.

First, we have to convert the data into a binary format which is easier for us to work with. VCF and Hapmap files are supported as input.

For this example, we assume that the training population is in a file called `train.hmp.txt` and the target population is in a separate file called `test.hmp.txt`. The file `train-phenotypes.csv` is a csv file which contains phenotypes for the training population. The first column is the taxa name (these names must match the column names in `train.hmp.txt`) and we are going to select the phenotype column called `yield`. All of these files are at the location `/path/to/my/data`.

First, let's do the training data:

```console
foo@bar:~$ docker run --rm -it -v /path/to/my/data:/data jubbens/gpfn --genotype_file train.hmp.txt --phenotype_file train-phenotypes.csv --phenotype_name yield

Running in docker detected.
CUDA is not available.

Calling the variants2bin script like this:
python /app/parsing/variants2bin.py --docker --genotype_file train.hmp.txt --phenotype_file train-phenotypes.csv --phenotype_name yield
Genotype path is /data/train.hmp.txt
Phenotype path is /data/train-phenotypes.csv
It looks like the genotype file is hapmap.
I'm going to try reading the genotype file...✔
I'm going to try reading the phenotype file...✔
Okay, let's parse the genotype file.
Trying to detect where the taxa start in the genotype file.
Grabbing the name of the first taxa from the phenotypes file, I assume it's in the first column.
Detected first col: 11
That means the first taxa is TRAIN-ONE
Reading the alleles column to figure out the encoding. I'm going to assume it's something like A/B...✔
I'm going to try and see if the sites are encoded with one or two characters.
Looks like sites are encoded with a single character.
Filtering the list of phenotypes to coerce missing ones to NaN...✔
Using all of the taxa names from the phenotype file.
Compiling data now, this may take a while...✔
Skipped due to missing genotype: 0
Skipped due to missing phenotype: 15
Total samples: 758
Finishing up and writing to disk...✔
Output file was written at /data/yield.variants2.bin 🍺
```

Next, let's do the test data:

```console
foo@bar:~$ docker run --rm -it -v /path/to/my/data:/data jubbens/gpfn --genotype_file test.hmp.txt

Running in docker detected.
CUDA is not available.

Calling the variants2bin script like this:
python /app/parsing/variants2bin.py --docker --genotype_file test.hmp.txt
No phenotype file provided, will not include any phenotype information.
Genotype path is /data/test.hmp.txt
It looks like the genotype file is hapmap.
I'm going to try reading the genotype file...✔
Okay, let's parse the genotype file.
Trying to detect where the taxa start in the genotype file.
Detected first col: 11
That means the first taxa is TEST-ONE
Reading the alleles column to figure out the encoding. I'm going to assume it's something like A/B...✔
I'm going to try and see if the sites are encoded with one or two characters.
Looks like sites are encoded with a single character.
Using all of the taxa from the genotype file.
Compiling data now, this may take a while...✔
Skipped due to missing genotype: 0
Skipped due to missing phenotype: 0
Total samples: 374
Finishing up and writing to disk...✔
Output file was written at /data/no-phenotypes.variants2.bin 🍺
```

You should now have two new files in your data directory: one containing the training population called `yield.variants2.bin` and another one with the target population called `no-phenotypes.variants2.bin`. Don't bother trying to open them as they're not human-readable.

### Step 2: Do Inference

We can now get predictions for our target population. Since we want to use the GPU for inference, let's include `--gpus all`. The only other thing to note is that we pointed to `deploy/pika.pt` as the model to use. A full list of available models is provided in the table above.

```console
foo@bar:~$ docker run --gpus all --rm -it -v /path/to/my/data:/data jubbens/gpfn --model_path deploy/mongoose.pt --train_path yield.variants2.bin --target_path no-phenotypes.variants2.bin

Running in docker detected.
CUDA is available.

Calling the inference script like this:
python /app/evaluate/inference.py --docker --model_path deploy/pika.pt --train_path yield.variants2.bin --target_path no-phenotypes.variants2.bin
Training samples: 758
Eval samples: 374

Success! Taxa and their predictions for the target dataset were written to /data/yield.variants2.bin-predicted.csv 🍺
```

You should be able to find the predicted GEBVs in `yield.variants2.bin-predicted.csv` in your data folder. Note that these predicted values are normalized - if the absolute values matter to you, then add the training population mean and multiply by its standard deviation.

## Fitting Models in Python

If you want to go a bit deeper and develop your own GPFNs, you can get your hands dirty with the Python code.

### Installation

Clone this repo.

```console
foo@bar:~$ git clone https://github.com/jubbens/gpfn.git
foo@bar:~$ cd gpfn
```

Dependencies are managed with conda, so you will need to make a new environment.

```console
foo@bar:~$ conda env create -f environment.yml
foo@bar:~$ conda activate gpfn
```

Make sure everything is good to go by running the test suite (optional).

```console
foo@bar:~$ pytest -v tests
```

### Fitting New Models Locally

It is possible to fit a GPFN using your local machine. This will realistically require at minimum a CUDA-capable GPU with at least 12 GB of VRAM as well as at least 8GB of RAM plus another 2-3 GB of RAM per CPU core. It's technically possible to run the fitting on CPU, but your CPU resources will be very over-subscribed since they will be busy drawing from the prior. 

```console
foo@bar:~$ python experiments/train_model.py --prior nam --num_threads 8
```

The only required command line argument is `--prior`, which will determine which prior to fit. Make sure you also specify the number of threads, which should correspond to the number of available CPU cores. Sensible defaults are chosen for the other arguments, but there is a lot of flexibility if you want to play around with the fitting process:

```console
foo@bar:~$ python experiments/train_model.py -h
usage: train_model.py [-h] --prior {wild,nam,silva,recurrent,ril} [--device DEVICE] [--use_cache] [--cache_dir CACHE_DIR] [--num_threads NUM_THREADS] [--feature_max_size FEATURE_MAX_SIZE] [--num_samples NUM_SAMPLES] [--pop_max_size POP_MAX_SIZE]
                      [--feature_selection {greedy,correlation,mi,sequential,regression,rp,g,composite,pca,strided_pca,downsample}] [--num_layers NUM_LAYERS] [--hidden_dim HIDDEN_DIM] [--num_heads NUM_HEADS] [--no_input_norm] [--emb_size EMB_SIZE] [--dropout_p DROPOUT_P]
                      [--min_training_samples MIN_TRAINING_SAMPLES] [--max_training_samples MAX_TRAINING_SAMPLES] [--nonuniform_eval_pos] [--batch_size BATCH_SIZE] [--lr LR] [--warmup_steps WARMUP_STEPS] [--accumulation_steps ACCUMULATION_STEPS] [--loss_function {bar,bar_mse}]
                      [--allow_tf32] [--num_buckets NUM_BUCKETS] [--save_path SAVE_PATH] [--do_log] [--log_steps LOG_STEPS] [--checkpoint_steps CHECKPOINT_STEPS] [--asynchronous]

Train a meta-learner on various different priors

optional arguments:
  -h, --help            show this help message and exit
  --prior {wild,nam,silva,recurrent,ril}
                        Which prior to fit (required) (default: None)
  --device DEVICE       Force a device to use (default: None)
  --use_cache           Use a cached dataset, or output cached dataset after generating (default: False)
  --cache_dir CACHE_DIR
                        The location to find or put cache files (default: /tmp)
  --num_threads NUM_THREADS
                        Number of simultaneous simulations. (default: 10)
  --feature_max_size FEATURE_MAX_SIZE
                        Number of features for feature selection (default: 100)
  --num_samples NUM_SAMPLES
                        Total samples to output (default: 400000)
  --pop_max_size POP_MAX_SIZE
                        Individuals per population (default: 1000)
  --feature_selection {greedy,correlation,mi,sequential,regression,rp,g,composite,pca,strided_pca,downsample}
                        Strategy for feature selection (default: pca)
  --num_layers NUM_LAYERS
                        Number of layers in transformer (default: 12)
  --hidden_dim HIDDEN_DIM
                        Size of fc layers in transformer (default: 2048)
  --num_heads NUM_HEADS
                        Number of heads for multi-head attention (default: 1)
  --no_input_norm       Forego input normalization (default: False)
  --emb_size EMB_SIZE   Dimensionality of input embedding (default: 2048)
  --dropout_p DROPOUT_P
                        Dropout rate for the transformer. (default: 0.0)
  --min_training_samples MIN_TRAINING_SAMPLES
                        Minimum samples for training. If not supplied, will use half the total samples. (default: None)
  --max_training_samples MAX_TRAINING_SAMPLES
                        Maximum samples for training. If not supplied, will use the total samples minus ten. (default: None)
  --nonuniform_eval_pos
                        Bias sampling towards larger train sets. (default: False)
  --batch_size BATCH_SIZE
                        Samples per batch (default: 8)
  --lr LR               Learning rate (default: 5e-05)
  --warmup_steps WARMUP_STEPS
                        Warmup steps for optimizer (default: 25000)
  --accumulation_steps ACCUMULATION_STEPS
                        Number of steps to accumulate gradients (default: 16)
  --loss_function {bar,bar_mse}
                        Loss function to use (default: bar_mse)
  --allow_tf32          Allow Tensor-Float32 precision for cuda matmul ops (default: False)
  --num_buckets NUM_BUCKETS
                        Number of buckets for discretizing output (default: 50)
  --save_path SAVE_PATH
                        Location to save the checkpoints and final model. (default: None)
  --do_log              Output to wandb (default: False)
  --log_steps LOG_STEPS
                        Log every n steps (default: 100)
  --checkpoint_steps CHECKPOINT_STEPS
                        Save a checkpoint every n steps (default: None)
  --asynchronous        If set, the dataloader watches the cache directory and waits for data to appear. (default: False)
```

The default batch size is based on a single 16GB GPU, so you may have to go with a smaller one. Of course, the number of gradient accumulation steps and/or the learning rate will need to be tuned.

### Fitting New Models on HPC

While it is technically possible to perform fitting on your local machine, GPFNs require many draws from the prior, and this involves a lot of simulation work. A single machine rarely has enough CPU cores to fit a GPFN in a sensible amount of time. For this reason, it is more practical to use a compute cluster.

The following are three ways to do this, in order from "*simplest but worst*" to "*most complicated but best*".

#### Option 1: Generating a Cached Dataset on a Single Node

The first approach is to use one cluster node to generate all the data and save it to disk, called a cache, and then afterward fit the GPFN using this cached dataset. Inside each prior's folder in `experiments`, you will find an example script for doing this called `generate_[prior].py`. First, create the dataset. Here is an example of doing this on a cluster using the [CVMFS](https://cvmfs.readthedocs.io/en/stable/) package manager and [Slurm](https://slurm.schedmd.com/documentation.html) scheduler. For this technique, you can use the conda environment just like if you were running locally. The following shows running on a single node with 56 CPU cores.

```sh
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-cpu=3000M

module load nixpkgs
module load python
module load anaconda3
conda activate gpfn

export PYTHONPATH=$PYTHONPATH:/my/home/directory/gpfn

srun python experiments/nam/generate_nam.py --output_dir /my/home/directory/cache --num_threads 56
```

You can then start a new job to fit the model on the cluster, or transfer the cache directory to your local machine to fit it there. Use the ``--use_cache`` and ``--cache_directory`` flags to use this cached dataset instead of generating new draws from the prior on the fly.

```console
foo@bar:~$ python experiments/train_model.py --prior nam --num_threads 8 --use_cache --cache_directory /my/home/directory/cache
```

#### Option 2: Generating a Cached Dataset on Multiple Nodes with MPI

Instead of using one node, you can also parallelize the process across multiple nodes using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). This requires that your cluster has an MPI implementation and is compatible with [MPI4Py](https://mpi4py.readthedocs.io/en/stable/). Because of some incompatibilities with the software stack, this needs to be done with a virtual environment instead of conda. This example parallelizes to 560 cores, which should saturate ten 56-core nodes. Well, that's sort of true. The first core will be used to coordinate the workers.

```sh
#!/bin/bash
#SBATCH --ntasks=560
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000M

module purge
module load StdEnv/2020
module load openmpi
module load mpi4py

source /my/home/directory/venvs/gpfn_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/my/home/directory/gpfn

srun python -m mpi4py.futures experiments/nam/generate_nam.py --mpi --output_dir /my/home/directory/cache

deactivate
```

#### Option 3: Generating Data with MPI and Fitting in Parallel

One downside of doing the two-part process of first generating data and then fitting the model is that the cached dataset may be extremely large, and may not fit on disk. This is typical for datasets with millions of draws. For this reason, the best option is to run one job to generate data and another in parallel to fit the model. This is accomplished with **asynchronous** mode. First, start the training script with the ``--asynchronous`` flag. Wait until it creates a metadata file in the cache directory called ``metadata.bin``. Then, start the same script as before to generate the data, pointing it to that same cache directory.

The generator script will see that metadata file and start generating data based on it. As the files are consumed by the training script, they will be deleted from the cache directory. This prevents the disk usage from growing, allowing you to generate large datasets on CPU nodes and consume the data simultaneously on a separate GPU job.

### Making New Priors

To create a new prior, two components need to be created: a **breeding scheme** and a **loader**. I recommend you look at `breeding_schemes/nam.py` and `loaders/nam_prior.py` for a working example.

A breeding scheme defines how a new population is created by crossing individuals from one generation to the next. The only requirement for a breeding scheme class is a `forward` function which returns two values, a training population and an evaluation population.

The corresponding loader extends the `Prior` superclass which itself is a Pytorch `Dataset`. Each call to `__getitem__` should return a tuple containing genotypes and phenotypes. The training and evaluation sets are combined here - the training script has other mechanisms to know where the train/test split is and will do that split when it receives the data.

## License

This material is licensed under the terms of the GPLv3 Open Source license.
