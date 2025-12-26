# Simple DDPM Diffusion Model (MNIST)

This repository contains a simple implementation of a **DDPM-style diffusion model** using PyTorch.

The model is trained on the MNIST dataset and learns to generate digit images by gradually removing noise from random inputs.

This project is designed for learning and practice purposes.

---

## What is implemented
- Forward diffusion process (adding Gaussian noise to images)
- A neural network trained to predict the added noise
- Reverse diffusion process to generate images from noise
- Training from scratch on the MNIST dataset
- Image sample generation and saving results

---

## Project structure
.
├── diffusion/
│   ├── simple_ddpm.py   # Main diffusion model code
│   └── notes.md         # Notes about the diffusion process
├── requirements.txt
├── samples.png          # Final generated samples
└── sample_epoch_1.png   # Sample during training
---

## How to run

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
2. Install required libraries
pip install -r requirements.txt
3. Train the model
python diffusion/simple_ddpm.py --train --epochs 5
During training, sample images will be saved after each epoch.
4. Generate samples
python diffusion/simple_ddpm.py --sample
After running this command, a file called samples.png will be created.
Output

The model generates digit images starting from random noise using the learned reverse diffusion process.

Example output can be found in samples.png.

⸻

Notes
	•	This is a simplified DDPM-style implementation.
	•	The focus is on understanding the diffusion process, not on performance optimization.
	•	The code is intentionally kept small and readable.
  Purpose

This project was implemented to gain hands-on experience with diffusion-based generative models and understand their core training and sampling mechanisms.
