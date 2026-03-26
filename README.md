## A Octave implementatio of HED

#### Introduction

Source implementation: https://github.com/xwjabc/hed.git

This is a PyTorch reimplementation of [Holistically-nested Edge Detection (HED)](https://arxiv.org/abs/1504.06375). The code is evaluated on Python 3.6 with PyTorch 1.0 (CUDA9, CUDNN7).

#### Instructions

##### Prepare

1. Download and extract the data:

   ```bash
   cd hed
   wget https://cseweb.ucsd.edu/~weijian/static/datasets/hed/hed-data.tar
   tar xvf ./hed-data.tar
   ```

##### Train and Evaluate

1. Train:

   ```bash
   python3 main.py --model HED --fine_tuning --dataset_name BSDS --dataset_folder [insert here]
   ```

   The results are in `output` folder. In the default settings, the HED model is trained for 40 epochs.

##### Evaluate my pretrained model

1. Evaluate the official pre-trained version:

   ```bash
   python main.py --checkpoint ./checkpoint/epoch-9-checkpoint.pt --output test_octave --model OCTHED --alpha 0.5 --test
   ```
   
#### Acknowledgement

This reimplementation is based on lots of prior works. Thanks to [Saining](https://github.com/s9xie/hed) for the original Caffe implementation. Thanks to [@meteorshowers](https://github.com/meteorshowers/hed-pytorch) for a PyTorch implementation where I adopt most of the code from. Thanks to [@jmbuena](https://github.com/jmbuena/toolbox.badacost.public) for a fixed version of Piotr's Toolbox. Thanks to [Berkeley Institute for Data Science](https://github.com/BIDS/BSDS500) which provides a mirror of BSDS500 dataset (the original link to the dataset seems broken).
