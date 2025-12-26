# Generative Models – Replication Studies (Starter)

Minimal DDPM-style diffusion implementation on MNIST.

## Run
pip install -r requirements.txt
python3 diffusion/simple_ddpm.py --train --epochs 5
python3 diffusion/simple_ddpm.py --sample --ckpt diffusion_ddpm_mnist.pt
