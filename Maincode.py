import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Initialize Tkinter window
root = tk.Tk()
root.title("Alternate Fashion Recommend System")

# Load model and data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to open file dialog for image selection
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        upload_image(file_path)

# Function to upload and process the selected image
def upload_image(file_path):
    try:
        img = Image.open(file_path)
        img = img.resize((400, 400))
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img
        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list)
        show_recommendations(indices)
    except:
        label.config(text="Error occurred while processing the image")

# Function for feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Function for recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# Function to show recommended images
def show_recommendations(indices):
    for i in range(1, 6):  # Skip the first index which is the query image
        image_path = filenames[indices[0][i]]
        img = Image.open(image_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        images_label[i-1].config(image=img)
        images_label[i-1].image = img

# Create GUI widgets
browse_button = tk.Button(root, text="Browse", command=browse_image)
label = tk.Label(root, text="Select an image to see recommendations")
images_label = []
for i in range(5):
    img_label = tk.Label(root)
    images_label.append(img_label)

# Place GUI widgets using grid layout
browse_button.grid(row=0, column=0, padx=10, pady=10)
label.grid(row=1, column=0, padx=10, pady=10)
for i in range(5):
    images_label[i].grid(row=2, column=i, padx=10, pady=10)

root.mainloop()
