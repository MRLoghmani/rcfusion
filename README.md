# RCFusion
Original implementation of the paper "Recurrent Convolutional Fusion for RGB-D Object Recognition": https://arxiv.org/pdf/1806.01673.pdf

Requirements:
* Tensorflow 1.5.0
* CUDA 8.0
* Python 2.7
* See python requirements in requirements.txt
* Docker + nvidia-docker [optional]

Instructions:
1. Download dataset and parameteres (see link below) and extract them in directory <dataset_dir> and <params_dir>
[Skip to point (4) to run w/o docker]
2. To execute the code within a docker container, run ```docker build -t <container_name> .```
3. Start the container with ```docker run -it --runtime=nvidia -v <dataset_dir>:<dataset_dir> -v <params_dir>:<params_dir> <container_name> bash```
4. Run ```python tran_and_eval.py --dataset <dataset_dir> --weights <params_dir>```

Download links:
* Pre-processed RGB-D Object dataset: https://data.acin.tuwien.ac.at/index.php/s/YKZQmoRtWaAcU91
* Original RGB-D Object dataset (we recommend using the link above instead to have the exact same pre-training we used in our experiments): http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/
* ResNet-18 pre-trained weights for initialization: https://data.acin.tuwien.ac.at/index.php/s/SopUlaRyoS4ct2Y

Contributors:
* Mohammad Reza Loghmani - email: loghmani@acin.tuwien.ac.at
* Mirco Planamente - email: mirco.planamente@iit.it

Citation:
```
@ARTICLE{rcfusion, 
author={M. R. {Loghmani} and M. {Planamente} and B. {Caputo} and M. {Vincze}}, 
journal={IEEE Robotics and Automation Letters}, 
title={Recurrent Convolutional Fusion for RGB-D Object Recognition}, 
year={2019}, 
volume={4}, 
number={3}, 
pages={2878-2885}, 
doi={10.1109/LRA.2019.2921506}}
```
