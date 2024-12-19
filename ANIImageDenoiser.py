import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
import distutils


# Function to process images with individual timing
def denoise_and_enhance_image(image):
    """Applies various denoising and enhancement techniques to the input image.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        dict: Dictionary containing processed images and their processing times.
    """
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = {}

    # Original noisy image
    noisy_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results["Noisy Image"] = (noisy_image, 0)  # Processing time for original is 0

    # Denoising and enhancement techniques
    techniques = {
        "Gaussian Blur": lambda img: cv2.GaussianBlur(img, (7, 7), 1.5),
        "Median Blur": lambda img: cv2.medianBlur(img, 7),
        "Bilateral Filter": lambda img: cv2.bilateralFilter(
            img, 15, 75, 75
        ),  # Adjusted parameters for better results
        "Non-Local Means Denoised": lambda img: cv2.fastNlMeansDenoisingColored(
            img, None, 15, 15, 7, 21  # Higher strength for denoising
        ),
        # New Advanced Denoising Techniques
        "Wavelet Denoising": lambda img: wavelet_denoise(
            img
        ),  # Placeholder for wavelet denoising function
        "BM3D Denoising": lambda img: bm3d_denoise(
            img
        ),  # Placeholder for BM3D function
    }

    for title, method in techniques.items():
        start_time = time.time()
        processed_img = method(image)
        processing_time = time.time() - start_time
        results[title] = (
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
            processing_time,
        )

    # Sharpening
    start_time = time.time()
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(results["Bilateral Filter"][0], -1, sharpening_kernel)
    processing_time = time.time() - start_time
    results["Sharpened Image"] = (sharpened_img, processing_time)

    return results


# Placeholder for wavelet denoising function
def wavelet_denoise(image):
    # Wavelet-based denoising logic will go here (use PyWavelets or similar library)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


# Placeholder for BM3D denoising function
def bm3d_denoise(image):
    # BM3D denoising logic will go here (use BM3D algorithm from OpenCV or third-party library)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


# Streamlit App
st.set_page_config(page_title="ANI Image Denoiser", layout="wide")

st.title("🖼️ ANI Image Denoiser (Advanced Noise Interpolation)")
st.markdown(
    "This app allows you to denoise and enhance images with multiple methods, measure processing times, and download results. "
    "Developed by **[Rupam Nag](https://rupam.netlify.app/)**."
)

# Sidebar for image upload
with st.sidebar:
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )
    st.markdown("---")
    st.markdown("Developed by **[Rupam Nag](https://rupam.netlify.app/)**")

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image
    image = Image.open(uploaded_file)
    st.write("### Processing the image...")
    results = denoise_and_enhance_image(image)

    # Display results in a grid layout
    cols = st.columns(len(results))
    for idx, (title, (result_image, processing_time)) in enumerate(results.items()):
        with cols[idx]:
            st.write(f"**{title}**")
            st.image(result_image, caption=title, use_column_width=True)
            st.write(f"⏱️ **Processing Time:** {processing_time:.2f} seconds")

            # Convert the image to downloadable format
            result_pil = Image.fromarray(result_image)
            buffer = io.BytesIO()
            result_pil.save(buffer, format="JPEG")
            buffer.seek(0)

            # Calculate file size
            file_size_bytes = len(buffer.getvalue())
            file_size_kb = file_size_bytes / 1024  # Convert to KB
            st.write(f"📁 **File Size:** {file_size_kb:.2f} KB")

            # Download button for each image
            st.download_button(
                label="Download",
                data=buffer,
                file_name=f"{title.replace(' ', '_').lower()}.jpg",
                mime="image/jpeg",
            )
else:
    st.write("👈 Upload an image from the sidebar to get started!")

# Footer
st.markdown("---")
st.markdown(
    "Developed by **[Rupam Nag](https://rupam.netlify.app/)** | Powered by [OpenCV](https://opencv.org/) and [Streamlit](https://streamlit.io/)"
)
