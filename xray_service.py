import torch
import torchxrayvision as xrv
import skimage.io
import pydicom
import numpy as np
from PIL import Image
import time
import os
from captum.attr import GradientShap, Saliency, IntegratedGradients

class XRayModel:
    def __init__(self):
        self.model = None
        self.transform = None
        
    def load_model(self):
        print("Loading TorchXRayVision model...")
        # Using densenet121-res224-all as requested
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval()
        print("Model loaded successfully.")
        
    def preprocess(self, file_path):
        start_time = time.time()
        
        # Handle DICOM
        if file_path.lower().endswith('.dcm'):
            ds = pydicom.dcmread(file_path)
            # De-identify: we are just reading pixel data, so effectively stripping metadata
            img = ds.pixel_array
        else:
            # Handle JPG/PNG
            img = skimage.io.imread(file_path)
            
        # Normalize to -1024, 1024 range as expected by xrv
        img = xrv.datasets.normalize(img, 255) 
        
        # Check channels
        if len(img.shape) > 2:
            img = img.mean(2) # Convert to grayscale
            
        # Add batch and channel dimensions: [1, 1, H, W]
        img = img[None, None, ...]
        
        # Resize if needed (xrv models often handle this, but for Grad-CAM we want specific size)
        # For simplicity relying on xrv transformation if manual resize needed:
        # transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        # But here we pass raw image to model if it handles, or resize manually.
        # xrv models expect 224x224 usually.
        import torch.nn.functional as F
        img_tensor = torch.from_numpy(img)
        img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        preprocess_time = time.time() - start_time
        return img_tensor, preprocess_time

    def predict(self, img_tensor):
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        inference_time = time.time() - start_time
            
        # Probabilities (sigmoid)
        probs = torch.sigmoid(outputs).detach().numpy()[0]
        
        # Top predictions
        results = []
        for i, label in enumerate(self.model.pathologies):
            results.append({
                "label": label,
                "prob": round(float(probs[i]) * 100, 2)
            })
            
        # Sort by probability
        results.sort(key=lambda x: x['prob'], reverse=True)
        return results, inference_time

    def explain(self, img_tensor, target_idx=0):
        # Using Saliency or GradientShap for explanation
        # For Grad-CAM style, we hook into features. 
        # Using Captum Saliency for simplicity/robustness on general CNNs
        
        saliency = Saliency(self.model)
        # We need gradients
        self.model.zero_grad()
        # target_idx is the class index we want to explain (e.g., highest prob)
        
        # Note: xrv models output raw logits usually, so Saliency works on that
        # img_tensor requires grad for captum
        img_tensor.requires_grad = True
        
        attr = saliency.attribute(img_tensor, target=target_idx)
        
        # Convert to heatmap
        heatmap = attr.detach().numpy()[0][0] # [1, 1, 224, 224] -> [224, 224]
        
        # Normalize heatmap for visualization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        
        # Colorize (apply simple colormap or just return gray for frontend to overlay)
        # Using matplotlib to apply 'jet' colormap and save
        import matplotlib.pyplot as plt
        
        # Create a unique filename
        heatmap_filename = f"heatmap_{int(time.time())}.png"
        save_path = os.path.join('static', 'heatmaps', heatmap_filename)
        
        plt.imsave(save_path, heatmap, cmap='jet')
        return save_path

# Global instance
xray_service = XRayModel()
