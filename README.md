To train model:

WaveGrad:
!python3 distillate.py
--module 
wavegrad
--diffusion 
WaveGradDiffusion
--name 
wavegrad
--dname 
base_0
--base_checkpoint 
path_to_checkpoint
--dataset 
path_to_dataset
--batch_size 
8
--num_workers 
4
--num_iters 
40000
--log_interval 
10
--params_type 
wavegrad
--time_scale
0.5
--n_timesteps 
1024

DiffWave:
!python3  distillate.py
--module
diffwave
--diffusion
GaussianDiffusionDefault
--name
diffwave
--dname
base_0
--base_checkpoint
/Users/kostiks/diffwave-ljspeech-22kHz-1000578.pt
--batch_size
1
--num_workers
4
--num_iters
5000
--log_interval
5
--dataset
/Users/kostiks/ljspeech/train
