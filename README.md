# SimCLR4Paint
SimCLR4Paint is experimental project that run training with illustration data.  
Now training is not yet ended, so my GPU is still working hard.

Training data is not yet shared now, but I'll share if training process is completed without any problems.  
Most of training data comes from danbooru dataset, so you can use danbooru2019 as training data.

## Dependency
This project may depend on following packages.

- pytorch
- pytorch-lightning
- tqdm
- jupyterlab
- Pillow

If you met errors because of the packages, please install missing packages.

## Usage
There are two way to run training process.

- main.py
- training-pl.ipynb

Both script used pytorch-lightning because of its usefulness and reproducibility.

### main.py
execute following command

```
$ python main.py --train_path path/to/train_data --valid_path path/to/valid_data 
```

Description of argument

```
--seed            default: 42             seed value
--batch_size      default: 32             batch size 
--epochs          default: 10             number of epochs
--projection_dim  default: 256            output feature size
--img_size        default: 512            input image size
--temperature     default: 0.5            hyperparameter used in loss
--train_path      default: "./data/train" path of training data
--valid_path      default: "./data/valid" path of validation data
```

### training-pl.ipynb
Launch jupyter notebook or jupyter lab and open notebooks/traing-pl.ipynb, then run all cells.

## Architecture
Model has pre-trained ResNet18 body for encoding and two layer dense nn for projecting h to z described in paper.  
Pre-traind model has trained with danbooru2018 dataset and is shared in [here](https://github.com/RF5/danbooru-pretrained/)

## References
SimCLR - [arxiv](https://arxiv.org/abs/2002.05709)

The Illustrated SimCLR Framework - [article](https://amitness.com/2020/03/illustrated-simclr/)

Understanding SimCLR - [Medium](https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c)

Other SimCLR implementation
- Spijkervet - [github](https://github.com/Spijkervet/SimCLR)
- sthalles - [github](https://github.com/sthalles/SimCLR)
