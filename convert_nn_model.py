"""
Convert Neural Network model from .pt to .pkl format
"""
import torch
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the .pt file
pt_path = os.path.join(BASE_DIR, 'models', 'neural network', 'neural_network_model.pt')
pkl_path = os.path.join(BASE_DIR, 'models', 'neural network', 'neural_network_model.pkl')

print(f"Loading from: {pt_path}")
checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

print(f"Saving to: {pkl_path}")
joblib.dump(checkpoint, pkl_path)

print("✓ Conversion complete!")
print(f"  state_dict keys: {checkpoint['state_dict'].keys()}")
print(f"  n_features: {checkpoint['n_features']}")
print(f"  R²: {checkpoint['r2']:.4f}")
