import streamlit as st
import cv2
import numpy as np
from PIL import Image

def load_image(image_file):
    return np.array(Image.open(image_file))

def sharpen_image(image, intensity):
    """
    Sharpens the image by blending it with its Laplacian edge map.
    The 'intensity' slider controls the blend strength.
    """
    # Convert to float32 for precise operations
    img_float = image.astype(np.float32)

    # Apply Laplacian to get edges
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F)

    # Blend the original with edges to enhance detail
    sharpened = img_float + intensity * laplacian

    # Clip values to valid range and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def smoothen_image(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def make_image_blue(image):
    blue_image = image.copy()
    blue_image[:, :, 0] = 0  # Remove red
    blue_image[:, :, 1] = 0  # Remove green
    return blue_image

def add_noise(image, sigma):
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

def rotate_image(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width//2, height//2), angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))

def canny_edge(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, threshold1, threshold2)

# Streamlit UI
st.title("🖼️ Interactive Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    st.markdown("---")
    operation = st.selectbox("Choose an operation to perform:", [
        "Sharpen", "Smoothen", "Enhance",
        "Make Blue", "Add Noise",
        "Rotate", "Canny Edge Detection"
    ])

    # Options for each operation
    if operation == "Sharpen":
        intensity = st.slider("Sharpening Intensity", 0.0, 2.0, 1.0, step=0.1)
        result = sharpen_image(image, intensity)


    elif operation == "Smoothen":
        k = st.slider("Kernel Size (odd number)", 1, 39, 7, step=2)
        result = smoothen_image(image, k)

    elif operation == "Enhance":
        result = enhance_image(image)

    elif operation == "Make Blue":
        result = make_image_blue(image)

    elif operation == "Add Noise":
        sigma = st.slider("Noise Intensity (σ)", 0, 100, 25)
        result = add_noise(image, sigma)

    elif operation == "Rotate":
        angle = st.slider("Rotation Angle", -180, 180, 45)
        result = rotate_image(image, angle)

    elif operation == "Canny Edge Detection":
        t = st.slider("Edge Threshold", 0, 150, 50)
        result = canny_edge(image, t, t * 2)

    # Show processed image
    st.image(result, caption=f"{operation} Result", use_column_width=True)
