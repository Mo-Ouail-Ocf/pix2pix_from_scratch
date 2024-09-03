import torch
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ignite.handlers import Checkpoint
from ignite.engine import Engine
from model import Generator

def load_checkpoint(checkpoint_dir, device='cuda'):
    checkpoint_path = Path(checkpoint_dir)
    
    gen = Generator().to(device)
    
    checkpoint = torch.load(checkpoint_path / 'model_gen_checkpoint.pth', map_location=device)
    Checkpoint.load_objects(to_load={'gen': gen}, checkpoint=checkpoint)
    
    return gen

def preprocess_image(image_path, transform, device='cuda'):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    # For validation data
    # input_img = img[:, :600, :]  
    input_img = transform(image=img)['image']
    input_img = input_img.unsqueeze(0).to(device)  
    
    return input_img

def postprocess_output(output_tensor, output_path):
    """Post-process the output tensor and save the image."""
    output_tensor = output_tensor.squeeze(0).cpu().detach()
    output_tensor = (output_tensor * 0.5 + 0.5) * 255.0  # Denormalize to [0, 255]
    output_tensor = output_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    
    output_img = Image.fromarray(output_tensor)
    output_img.save(output_path)

def infer(model, input_image):
    model.eval()
    with torch.no_grad():
        generated_img = model(input_image)
    return generated_img

def main(image_path, checkpoint_dir, output_path, device='cuda'):
    # Load the model
    gen = load_checkpoint(checkpoint_dir, device=device)
    
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ]
    )
    
    input_image = preprocess_image(image_path, transform, device=device)
    

    generated_img = infer(gen, input_image)
    
    postprocess_output(generated_img, output_path)


if __name__ == "__main__":
    image_path = 'path/to/input_image.jpg'
    checkpoint_dir = 'pix2pix_models'
    output_path = 'path/to/your/output_image.jpg'
    
    main(image_path, checkpoint_dir, output_path, device='cuda')
