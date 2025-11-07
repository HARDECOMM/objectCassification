# Streamlit app for image classification
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import os
from torchvision import transforms
import numpy as np

# Define the modifiedResnet class (same as in training)
class modifiedResnet(nn.Module):
  def __init__(self):
    super(modifiedResnet, self).__init__()
    self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in self.resnet.parameters():
      param.requires_grad = False
    num_ftrs = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(num_ftrs, 6) # Assuming 6 classes

  def forward(self, x):
    return self.resnet(x)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = modifiedResnet().to(device)

# Define the path to the saved model file
model_path = 'image_classification_model.pth'

# Load the state dictionary
@st.cache_resource # Cache the model loading
def load_model(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model(model_path, device)

# Define the transformation for inference (should match validation/test transforms)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the list of class names in the correct order
# Make sure this order matches the label encoding in the dataset class
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(image):
    """
    Preprocesses an image and predicts the class label using the loaded model.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        str: The predicted class name or None if an error occurs.
    """
    try:
        # Ensure image is in RGB format before transforming
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Apply transformations
        img_transformed = inference_transforms(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(img_transformed)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = class_names[predicted_idx.item()]

        return predicted_class

    except Exception as e:
        st.error(f"Error during image prediction: {e}")
        return None

# Streamlit App UI
st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Predict the class
        predicted_class = predict_image(image)

        if predicted_class:
            st.success(f'Prediction: {predicted_class}')
        else:
            st.error('Could not classify the image.')

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
