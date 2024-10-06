# Teaching-to-pack

**This project is aim to explpore the efforts of deploying diffusion policy on aloha robots.
Mainly focusing on CNN-based and transformer-based diffusion policy.**

![Demo GIF](/mnt/d/kit/ALR/dataset/test_david/test_results/denoising_step_0.gif)

# Installation 

environment creation: 
```console
conda env create -f env_ttp.yaml
```
/Teaching_to_pack/src/aloha/: 
```console
git clone https://github.com/tonyzhaozh/aloha.git
```
/Teaching_to_pack/src/d3il_david/: 
```console
git clone https://github.com/joeyfreakman/d3il_david.git 
git checkout real_robot
``` 
# Usage

## File system

Teaching_to_pack
├── environment # 
│   └── data # data saving folder 
│   └── dataset # experiments of different datasets and dataloader which decrease I/O 
│   ...
├── scripts    #functions of dealing with data
│       ├── data_compressor    # code for compressing raw hdf5 data
│       ├── data_pruning       # apply modifications on collected images
│       ├── spatial position      # plot spatial trajectory of end effector
│       ...
├── src # model and policy
│   ├── aloha # aloha robots functions and data-collection and hyperparameters
│   ├── config # task configs and model hyper parameters
│   ├── core  # code for running the model
│   ├── d3il_david #transformer-based diffusion, usage refering to https://github.com/joeyfreakman/d3il_david/blob/real_robot/README.md
│       ...
│   ├──model # vision encoder backbone and some support-functions
│   ├──policy # cnn-based diffusion policy matching with different format datasets

## run cnn-based policy 

modify the train.sh file to run the file u need
```console
cd ~/Teaching_to_pack/src/core
bash train.sh 
```

## run diffusion-based policy

other modifications follow the instructions of d3il repository
```console
cd ~/Teaching_to_pack/src/d3il_david
python run.py
```