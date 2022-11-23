# Mixture of Domain-specific Experts
## Introduction
The vision based end-to-end autonomous driving framework for CARLA 0.8 benchmarks.
A pytorch implementation for Mixture of Domain-specific Experts (MoDE) framework ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/20000)).

This repository contains the following modules
 1. Disentanglement_VAE: Disentangling domain-specific feature and domain-general feature from pair images using Cycle-consistent VAE.
 2. ACTION_MINE: Prediction action values to control an ego-vehicle using representation learning and mixture of experts model.
 
## Getting Started

### Dependencies
* Major dependencies
  1. Python 3.7
  2. Pytorch 1.6
  3. cuda 10.2
 
### Installing
* Importing an uploaded Anaconda environment (torch.yaml) is recommended.

### Database Acquisition
* Method for acquisition of driving data on CARLA simulator is described in this [repository](https://github.com/carla-simulator/data-collector).

### CARLA Simulator and Benchmarks
* You can download from this [document](https://carla.org/2018/04/23/release-0.8.2/).

### Executing program
* First Stage: Training cycle-consistent VAE 
 1. Collecting pair images using the CARLA datacollector
 2. Move to the "Disentanglement_VAE"
 3. Modify the database path variables (train_pair, eval_pair) in train_CycleVAE_lusr_v2.py 
 4. Run the script using below command
```
python train_CycleVAE_lusr_v2.py --id="ID for this training"
```
  (Download a pre-trained weight file from [here](https://drive.google.com/file/d/1RtiwGAgRMl5Lpd5fyAA7cQbODWOIBqD6/view?usp=sharing))
 
 5. The trained weights are saved at save_models/id/id.pth
 

* Second Stage: Training autonomous driving framework
 1. Collecting driving dataset using CARLA datacollector
 2. Mode to the "ACTION_MINE"
 3. Run the script using below command
```
python main_wo_weatmask_posi_50_v2_gating.py --id="ID for this training" --train-dir="Training Dataset Path" --eval-dir="Evaluating Dataset Path" --vae-model-dir="Weight path trained by train_CycleVAE_lusr_v2.py"
```
  (Download a pre-trained weight file from [here](https://drive.google.com/file/d/1TyPY5pT7hANtXGsH5VAKauxdWkBLLUrI/view?usp=sharing))
 
4. Evaluating using the CARLA benchmark

* Third Stage: Run Benchmark
 1. Go to the CARLA 0.8.X folder
 2. Run the CARLA simulator
 ```
(Town01) sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000  -benchmark -fps=10 -ResX=800 -ResY=600
(Town02) sh CarlaUE4.sh /Game/Maps/Town02 -windowed -world-port=2000  -benchmark -fps=10 -ResX=800 -ResY=600
 * You can change the parameters according to the your experimental conditions.
```

 3. Run an evaluation script in 'driving-benchmark-AAAI' using below command
```
python run_representation_action_mine_posi50_gating.py --corl-2017 (or --carla100) --continue-experiment --model-path='Weight path trained by main_wo_weatmask_posi_50_v2_gating.py' --vae-model-dir="Weight path trained by train_CycleVAE_lusr_v2.py"
```
