# Import necessary libraries
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation

# Import ADE_CLASSES from ade module
from ade import ADE_CLASSES

# Uncomment the following line if you want to manually set ADE_CLASSES
# ADE_CLASSES = ['cat', 'dog', 'puppy', 'kitten']

# Check if CUDA is available, if not use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model name
model_name = 'nvidia/segformer-b5-finetuned-ade-640-640'

#  AutoFeatureExtractor gets required pre-processing steps for input data for the loaded model architecture
# and then apply those steps when invoked on that data. It has complex pipeline for matching the model name with all
# the saved model names and many other mandatory checks.
extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Load the Segformer model for semantic segmentation
model = SegformerForSemanticSegmentation.from_pretrained(model_name, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16).to(device)

# Get the class labels from the model configuration
classes = model.config.id2label

# Main execution
if __name__ == '__main__':

    # Define the path to the image
    path = "sample_imgs/FB_IMG_1694996981866.jpg"

    # Read the image using OpenCV
    image = cv2.imread(path)

    # Define the classes and mask options
    CLASSES = ADE_CLASSES
    mask_option = 'dilate'
    dilation_kernel = (9, 9)
    iterations = 0

    # Start the timer
    t0 = time.time()

    # Preprocess the image and convert it to tensor
    inputs = extractor(images=image, return_tensors="pt").to(device)

    # Run the model for 100 iterations
    for i in range(100):
        outputs = model(**inputs)

    # Interpolate the output logits to match the original image size
    logits = nn.functional.interpolate(outputs.logits.detach().cpu(), size=image.shape[:-1], mode='bilinear', align_corners=False)

    # Stop the timer
    t1 = time.time()

    # Get the class with the highest probability for each pixel
    seg = logits.argmax(dim=1)[0]
    seg = seg.numpy()
    result = seg.copy()

    # Initialize lists to store class labels and indices
    class_labels = []
    class_idxs = []

    # Check if each class is present in the segmentation result
    for i, class_label in enumerate(CLASSES):
        class_idx = ADE_CLASSES.index(class_label)
        if np.any(result == class_idx):
            class_labels.append(class_label)
            class_idxs.append(class_idx)

    # Convert the segmentation result to binary mask
    for idx in class_idxs:
        result[result == idx] = 255
    result[result != 255] = 0

    # Convert the result to float32
    result = result.astype(np.float32)

    # Define the kernel for dilation/erosion
    kernel = np.ones(dilation_kernel, np.uint8)

    # Apply dilation or erosion to the mask
    if mask_option == 'erode':
        result = cv2.erode(result, kernel, iterations=iterations)
    else:
        result = cv2.dilate(result, kernel, iterations=iterations)

    # Convert the result to an image
    mask = Image.fromarray(result.astype(np.uint8))

    # Calculate the time taken
    time_taken = t1 - t0
    print(f'Time taken: {time_taken/100}')

    # Uncomment the following line to save the mask
    # mask.save('sample_imgs/test.png')

    # Display the mask
    mask.show()

# b=np.transpose(inputs['pixel_values'][0].detach().cpu().numpy(), (1,2,0))
# c=np.array(b, dtype=np.uint8)
