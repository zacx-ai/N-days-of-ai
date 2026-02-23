## Project: SRCNN Super-Resolution

This project trains a **SRCNN (Super-Resolution Convolutional Neural Network)** on the **DIV2K dataset** to enhance low-resolution images.

- `upscaling/srcnn_full.py` → Full training script
- The code includes dataset handling, SRCNN model, training and testing loops.

### Features

- Single-file implementation for easy tracking and commits
- Supports GPU acceleration if available
- Logs train and test loss per epoch

---

## Dataset

The project uses the **DIV2K High-Resolution dataset**.  

To download the dataset, you can use:

```python
import kagglehub
path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
