# Denoising Diffusion Probabilistic Models


This repo contains code for DDPM training. 
Based on [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 
[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) and (in future) on [Diffusion Models Beat GANs on Image Synthesis
](https://arxiv.org/abs/2105.05233).

---

## Usage 

1) Install dependencies with `pip install -r requirements.txt`.
2) To run training script with default configs, simply write `python diffusion_lightning` (default config: config/task/cifar10.yaml)

Command-line arguments provided with [Hydra](https://hydra.cc/) library. 

---
