# Lab 3: Variational Autoencoder (VAE)
## CSET419 – Introduction to Generative AI

---

## Objective
The objective of this lab is to understand and implement a **Variational Autoencoder (VAE)** to:
- Learn latent representations of data  
- Generate diverse and novel samples  
- Understand encoder, decoder, latent space, and KL-divergence  

---

## Dataset
**MNIST Dataset**  
Handwritten digit images (28×28 grayscale).

---

## Project Structure
```
VAE_Lab3/
│
├── venv/                   # Virtual environment
├── data/                   # Dataset (auto-downloaded)
│
├── dataset.py              # Dataset loading & preprocessing
├── model.py                # VAE architecture
├── train.py                # Training script
├── generate.py             # Sample generation
├── visualize.py            # Reconstruction visualization
│
├── requirements.txt        # Dependencies
├── README.md               # Lab documentation
```

---

## Virtual Environment Setup

### Create Virtual Environment
```bash
python -m venv venv
```

### Activate Virtual Environment (Windows)
```bash
venv\Scripts\activate
```

---

## Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Implementation Details

### Encoder
- Input: Flattened 28×28 image
- Outputs: Mean (μ) and Log Variance (log σ²)

### Reparameterization Trick
Allows backpropagation through stochastic sampling.

### Decoder
- Input: Latent vector
- Output: Reconstructed image

---

## Loss Function
Total Loss =  
- **Reconstruction Loss** (Binary Cross Entropy)  
- **KL Divergence Loss**

---

## Training
```bash
python train.py
```

- Trains VAE for 10 epochs
- Saves trained model as `vae.pth`

---

## Sample Generation
```bash
python generate.py
```

- Generates new handwritten digits from random latent vectors

---

## Reconstruction Visualization
```bash
python visualize.py
```

- Displays original vs reconstructed images

---

## Results
- Trained VAE model
- Reconstructed images
- Newly generated samples
- Loss printed per epoch


---

## Conclusion
This lab demonstrates how VAEs learn probabilistic latent spaces and generate new data samples, making them powerful generative models.

---

**Course:** CSET419 – Introduction to Generative AI  
**Lab:** 3 – Variational Autoencoder  
