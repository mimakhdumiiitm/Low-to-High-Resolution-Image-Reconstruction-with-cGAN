# Low-to-High-Resolution-Image-Reconstruction-with-cGAN

This repository contains code and resources developed for the **NPPE 2 (Proctored Exam 2)** of the **Introduction to Deep Learning and Generative AI** course at **IIT Madras**.

> 🔒 **Note:** Since this Kaggle competition has been made private by the authority, I am uploading and sharing my complete work and implementation here.

⚠️ **Notebook Note:** GitHub may not render the `.ipynb` file due to a persistent `nbformat`/`nbconvert` issue. A Google Colab link is provided below for viewing and execution.

## 🚀 Quick Links
- **Google Colab:** [Open Notebook in Colab](https://colab.research.google.com/drive/17Srm1kvUF4dr6BI3hD8meKSM-x6S6LvB?usp=sharing) 
- **Kaggle Challenge:** [Plant Leaves Super-Resolution Challenge](https://www.kaggle.com/competitions/plant-leaves-super-resolution-challenge)

---

## 🛠️ Project Architecture

The solution implements a Super-Resolution conditional Generative Adversarial Network (cGAN) based on the **ESRGAN** architecture with **RRDB (Residual-in-Residual Dense Blocks)** to upscale low-resolution leaf images ($32 \times 32$) to high-resolution images ($128 \times 128$).

1. **Generator (RRDB-lite / EDSR-style):**
   - Utilizes **Residual Dense Blocks (RDB)** layered into an **RRDB** pipeline to preserve complex textual details.
   - Upsampling is performed via **PixelShuffle** layers for artifact-free scaling.
2. **Discriminator:**
   - A **PatchGAN** structured architecture that classifies local $N \times N$ patches as real or fake, forcing sharp texture profiles.
3. **Loss Functions (MAE-Dominant Optimization):**
   - **Pixel Loss:** L1 Loss (Dominant weight $\Lambda_{L1} = 100$) to minimize Mean Absolute Error (MAE).
   - **Perceptual Loss:** Evaluated against intermediate feature maps of a pretrained **VGG19** network.
   - **Adversarial Loss:** Evaluated via **LSGAN (Least Squares GAN)** loss parameters for stable convergence.

---

## 📈 Key Configuration Details

```python
class Config:
    LR_SIZE         = 32
    HR_SIZE         = 128
    SCALE           = 4
    BATCH_SIZE      = 16
    NUM_EPOCHS      = 150
    LR_G            = 1e-4
    LR_D            = 1e-4
    LAMBDA_L1       = 100.0
    LAMBDA_ADV      = 1.0
    LAMBDA_PERC     = 10.0
    MIXED_PRECISION = True  # Automatic Mixed Precision (AMP) enabled
