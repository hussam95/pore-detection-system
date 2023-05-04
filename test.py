import torch
import torchvision.transforms as transforms
from PIL import Image
from model import PoreNet
import os
working_dir = os.getcwd()
# Load the saved model from disk
model = PoreNet()
model.load_state_dict(torch.load('porenet.pth'))
model.eval()

# Define the image transformations to match those used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prompt the user to enter the path to an image
img_path = working_dir + "\\Group D_D_0_16.bmp"

# Load the image and apply the transformations
img = Image.open(img_path).convert('L')
img_tensor = transform(img).unsqueeze(0)

# Process the image using the trained model and print the results
with torch.no_grad():
    output = model(img_tensor)
    avg_diameter, max_diameter = output.squeeze().tolist()
    print("Average pore diameter:", avg_diameter)
    print("Biggest pore diameter:", max_diameter)
