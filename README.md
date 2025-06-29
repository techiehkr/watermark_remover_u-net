# AI Watermark Remover

This project provides a deep learning-based AI tool to remove watermarks from images using a UNet model.

The app uses a combination of MSE and perceptual (LPIPS) losses for training, and provides a user-friendly Gradio web interface for inference.

---

## Features

- Removes watermarks from images with the help of a user-provided mask.
- Built with PyTorch and UNet architecture.
- Easy-to-use web interface using Gradio.
- Trained on a custom dataset of watermarked and clean images.

---

## How to run locally

1. Clone the repository

```bash
git clone https://github.com/techiehkr/watermark_remover_u-net.git
cd watermark_remover_u-net
```

```pip install -r requirements.txt```


```python app.py```

Open the local URL (usually http://127.0.0.1:7860) in your browser.

## Files
 - app.py: Gradio interface and inference code.
 - model.py: UNet model definition.
 - watermark_remover.pth: Pre-trained model weights.
 - requirements.txt: Python dependencies.