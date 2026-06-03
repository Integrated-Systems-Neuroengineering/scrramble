ScRRAMBLe: BLOCK-SPARSE DEEP LEARNING ARCHITECTURE FOR ANALOG IN-MEMORY COMPUTING ACCELERATORS
---
This repository contains code to reproduce results and train ScRRAMBLe.

## Getting Started

### Running Locally

You can clone this repository using either `git` or `gh` (GitHub CLI):

**Using Git:**
```bash
git clone https://github.com/Integrated-Systems-Neuroengineering/scrramble.git
cd scrramble
```

**Using GitHub CLI:**
```bash
gh repo clone Integrated-Systems-Neuroengineering/scrramble
cd scrramble
```

### Create Conda Environment

Create a conda environment with the required packages:

```bash
conda env create -f environment.yml -n scrramble_env
```

Activate the environment:

```bash
conda activate scrramble_env
```

### Install Package in Editable Mode

Install the package in editable mode so that models and utilities in `src` are accessible:

```bash
pip install -e .
```

This allows you to import the package modules directly:

```python
from src.models import ...
from src.utils import ...
```

Any changes made to the source code in `src` will be immediately reflected without reinstalling.

### Running in Google Colab

You can run ScRRAMBLe directly in Google Colab without installing anything locally to leverage Google's GPU/TPU acceleration. Add the following cells to your Colab notebook:

**Cell 1: Clone the repository and install dependencies**
```python
!git clone https://github.com/vikrant-github/scrramble.git
%cd scrramble
!conda env update -f environment.yml
```

**Cell 2: Install the package in editable mode**
```python
!pip install -e .
```

**Cell 3: Run training scripts**
```python
# For MNIST
!python3 scrramble_mnist_training.py --connection_density 0.2 --slot_size 64 --resample 1 --seed 101 --train_steps 30000

# For CIFAR-10
!python3 scrramblexresnet_cifar10_training.py --connection_density 0.2 --slot_size 16

# For CIFAR-100
!python3 scrramblexresnet_cifar100_training.py --connection_density 0.75 --slot_size 16 --capsule_sizes 20 10 --train_steps 50000
```

Note: Colab provides GPU/TPU acceleration by default, so no additional setup is needed. For best results, ensure GPU is enabled in Colab runtime settings (Runtime → Change runtime type → GPU).

### Testing the model on datasets:
The `Tutorials/` folder contains reference scripts to train the ScRRAMBLe and ScRRAMBLe-ResNet Models. The general structure of these scripts can be used to reproducte figures in the manuscript.
Scripts used to generate figures in the manuscript can also be found in the `Tests/` folder.

#### Hardware used for simulations
These simulations were performed on a server with one _NVIDIA L40S_ GPU. The `jax` backend for code automatically selects the available hardware for running these scripts, so no adjustments to the code are needed. While not necessary, it is recemmended to use GPU acceleration for running these scripts.

#### MNIST Tests
This code simulates the ScRRAMBLe-CapsNet framework to classify and reconstruct MNIST images. Reconstructed images can be optionally saved using the `plot_reconstructions` argument. Run the following to train the model (you can add more arguments as needed) 

```bash
python3 scrramble_mnist_training.py --connection_density 0.2 --slot_size 64 --resample 1 --seed 101 --train_steps 30000 --plot_reconstruction
```

#### CIFAR-10 Tests
This code simulates ScRRAMBLe-ResNet architecture to classify CIFAR-10 images. Run the following to train the model (you can add more arguments as needed).

```bash
python3 scrramblexresnet_cifar10_training.py --connection_density 0.2 --slot_size 16
```


#### CIFAR-100 Tests
This code simulates ScRRAMBLe-ResNet architecture to classify CIFAR-100 images. Run the following to train the model (you can add more arguments as needed).

```bash
python3 scrramblexresnet_cifar100_training.py --connection_density 0.75 --slot_size 16 --capsule_sizes 20 10 --train_steps 50000
```

