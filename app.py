import streamlit as st
import numpy as np
import os
import cv2

from hog_algorithm import compute_hog
from euclidean import find_similar

# Set page title
st.set_page_config(page_title="Multimedia Image.")

def color_extraction(image):
    image_tmp = cv2.resize(image, (96, 128))
    non_white_indices = np.mean(image_tmp, axis=2, dtype=np.float16) < 250.0
    non_white_image = image_tmp[non_white_indices]
    color_sum = np.sum(non_white_image, axis=0)
    color_weight = [(value / color_sum.sum()) * 100 for value in color_sum]
    image_color = np.average(non_white_image, axis=1, weights=color_weight)
    return image_color

# Define function to display image
def show_image(image, results):
    row_1 = st.columns(5)
    row_2 = st.columns(5)
    row_3 = st.columns(5)
    dir_ = os.getcwd()

    with row_1[2]:
        st.write(f"Origin Image", unsafe_allow_html=True, width=100, height=10)
        st.image(image, channels="BGR", use_column_width='always')
    for i in range(5):
        with row_2[int(i % 5)]:
            image_index = results.loc[i, 'id']
            for type_ in ['png', 'jpg']:
                image_path = os.path.join(dir_, f'images/train/I_{image_index}.{type_}')
                if os.path.exists(image_path):
                    break
            image = cv2.imread(image_path)
            st.write(f"{image_path.split('/')[-1]}", unsafe_allow_html=True, width=100, height=10)
            st.image(image, channels="BGR", use_column_width='always')
            # st.write(f"{results.loc[i, 'distance']}", unsafe_allow_html=True, width=100, height=10)
        with row_3[int(i % 5)]:
            image_index = results.loc[i + 5, 'id']
            for type_ in ['png', 'jpg']:
                image_path = os.path.join(dir_, f'images/train/I_{image_index}.{type_}')
                if os.path.exists(image_path):
                    break
            image = cv2.imread(image_path)
            st.write(f"{image_path.split('/')[-1]}", unsafe_allow_html=True, width=100, height=10)
            st.image(image, channels="BGR", use_column_width='always')
            # st.write(f"{results.loc[i + 5, 'distance']}", unsafe_allow_html=True, width=100, height=10)

# Create file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If file is uploaded
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    origin_image = cv2.imdecode(file_bytes, 1)

    fixed_size = (384, 512)
    image = cv2.resize(origin_image, fixed_size)

    image_color = color_extraction(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_hog = compute_hog(image)

    image_feature = np.concatenate([image_hog, image_color])
    results = find_similar(image_feature)
    results = results.sort_values(by=['distance']).reset_index(drop=True)
    results['distance'] = results['distance'].apply(lambda x: round(x, 3))
    # # Display image
    show_image(origin_image, results)
else:
    st.write("Please upload an image.")
