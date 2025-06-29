import gradio as gr
from PIL import Image
import torch
import numpy as np
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet()

model.load_state_dict(torch.load("watermark_remover.pth", map_location=device, weights_only=True))

model.to(device)
model.eval()

def predict(w_img, mask_img):
    from PIL import ImageOps

    
    if isinstance(w_img, np.ndarray):
        w_img = Image.fromarray(w_img.astype(np.uint8)).convert("RGB")

   
    if isinstance(mask_img, dict) and "composite" in mask_img:
        mask_array = mask_img["composite"]
        mask_img = Image.fromarray(mask_array.astype(np.uint8)).convert("L")
    elif isinstance(mask_img, np.ndarray):
        mask_img = Image.fromarray(mask_img.astype(np.uint8)).convert("L")

    # Resize both images
    w_img = w_img.resize((256, 256))
    mask_img = mask_img.resize((256, 256))

    # Normalize
    w_tensor = torch.tensor(np.array(w_img)).permute(2, 0, 1).unsqueeze(0) / 255.0
    m_tensor = torch.tensor(np.array(mask_img)).unsqueeze(0).unsqueeze(0) / 255.0

    # Predict
    input_tensor = torch.cat([w_tensor, m_tensor], dim=1).to(device)
    with torch.no_grad():
        out = model(input_tensor).squeeze().permute(1, 2, 0).cpu().numpy()

    return Image.fromarray((out * 255).astype(np.uint8))

gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Watermarked Image"),
        gr.Sketchpad(label="Mask (Draw over watermark)")
    ],
    outputs=gr.Image(label="Restored Image"),
    title="AI Watermark Remover"
).launch()
