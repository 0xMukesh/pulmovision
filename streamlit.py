import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Model
from constants import DATASET_CLASSES


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    return transform(image).unsqueeze(0)


def predict(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        # using sigmoid as chestMNIST is multi-label dataset
        probabilities = torch.sigmoid(output)
    return probabilities.cpu().squeeze().numpy()


st.title("notyetfiguredout")
st.write("upload an image of chest x-ray for analysis")

uploaded_file = st.file_uploader(
    "choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="image uploaded", use_container_width=True)

    if st.button("analyze"):
        with st.spinner("analyzing..."):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model = Model(in_channels=1, num_classes=14).to(device)

            try:
                checkpoint = torch.load("model.pth", map_location=device)

                model_state_dict = checkpoint["model_state_dict"]

                model = Model(in_channels=1, num_classes=14)
                model.load_state_dict(model_state_dict)

                processed_image = preprocess_image(image)
                preds = predict(processed_image, model, device)

                st.subheader("results:")

                significant_findings = [
                    (label, prob) for label, prob in zip(DATASET_CLASSES, preds) if prob > 0.1]

                for label, prob in sorted(significant_findings, key=lambda x: x[1], reverse=True):
                    st.write(f"{label}: {prob:.3%}")
                    st.progress(float(prob))

            except Exception as e:
                st.error(f"err loading model: {str(e)}")
