import cv2
#import dlib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_dominant_colors(image, k=5):
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Apply k-means clustering to find the top k colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_.astype(int)

    # Count the number of pixels in each cluster
    counts = np.bincount(kmeans.labels_)

    # Calculate the percentage of each color
    percentages = counts / len(kmeans.labels_)

    return colors, percentages

def plot_colors(colors, percentages):
    # Create a bar chart to display the colors and their percentages
    plt.figure(figsize=(8, 6))
    plt.pie(percentages, labels=[f'{int(p*100)}%' for p in percentages], colors=[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in colors], startangle=90)
    plt.axis('equal')
    plt.show()

def detect_face(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No faces detected in the image")

    # Assume the first detected face is the region of interest
    x, y, w, h = faces[0]
    face_region = image[y:y+h, x:x+w]

    # Plot face region of interest for debugging
    plt.imshow(face_region)
    plt.title('Face Region of Image')
    plt.axis('off')  # Hide the axis
    plt.show()

    return face_region

# Load the image
image_path = 'C:/Users/RexKong/Downloads/Photo Enhancement/Jpeg/IMG_7685.jpg'

# Detect the face region in the image
face_region = detect_face(image_path)

# Get the top 5 colors and their percentages
colors, percentages = get_dominant_colors(face_region, k=5)

# Print the results
for i, (color, percentage) in enumerate(zip(colors, percentages)):
    print(f"Color {i+1}: RGB={color}, Percentage={percentage:.2%}")

# Plot the colors
plot_colors(colors, percentages)