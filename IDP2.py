import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import vision
import tkinter as tk
from tkinter import filedialog
import re

# Set Google Cloud authentication key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\aishu\Downloads\IDP\intelligent-doc-pa-d284df1ce962.json"

# Set up Google Vision API
client = vision.ImageAnnotatorClient()

# Function to select image using GUI
def upload_image():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# Step 1: Upload an image
image_path = upload_image()

if not image_path:
    print("No image selected. Exiting...")
    exit()

# Step 2: Read the image using OpenCV
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Convert RGBA/PNG images to RGB if needed
if len(img.shape) == 3 and img.shape[-1] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding (black filter) to enhance text detection
_, black_filtered = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show processed image
plt.imshow(black_filtered, cmap='gray')
plt.title("Processed Image for OCR")
plt.show()

# Step 3: Convert image to Google Vision format
with open(image_path, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

# Step 4: Perform text detection using Google Vision API
response = client.text_detection(image=image)
texts = response.text_annotations

if not texts:
    print("No text detected! Exiting...")
    exit()

# Extract full detected text
full_text = texts[0].description
print("\nExtracted Text:\n", full_text)

# Step 5: Function to extract specific fields using regex
def extract_field(text, field_name):
    """Extracts specific field values from the OCR text using regex."""
    patterns = {
        "name": r"(?:Name|Given Name|Surname):\s*([A-Za-z\s]+)",
        "nationality": r"(?:Nationality|Country):\s*([A-Za-z\s]+)",
        "date of birth": r"(?:DOB|Date of Birth|Birth Date):\s*([\d\-\/]+)",
        "passport number": r"(?:Passport No|Passport Number):\s*(\w+)"
    }
    pattern = patterns.get(field_name.lower())
    if pattern:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else "Not found"
    return "Invalid field"

# Step 6: Prompt user for specific field extraction
query = input("\nEnter field to extract (Name, Nationality, Date of Birth, Passport Number): ").strip().lower()
result = extract_field(full_text, query)

# Step 7: Display extracted result
print(f"\nExtracted {query.capitalize()}: {result}")
