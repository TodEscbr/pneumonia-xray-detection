import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN()
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.eval()

st.title("Sistem Pengesanan Pneumonia (PyTorch)")
uploaded_file = st.file_uploader("Muat naik gambar X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar X-ray",  use_container_width=True)
    
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)[0].item()

    if prediction > 0.5:
        st.error("⚠️ Gambar menunjukkan tanda-tanda Pneumonia.")
    else:
        st.success("✅ Gambar menunjukkan tiada tanda Pneumonia.")
