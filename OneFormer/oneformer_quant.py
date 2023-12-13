# Import necessary libraries
import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
import torch.nn.functional as F
from collections import defaultdict
from matplotlib import cm
import matplotlib.patches as mpatches
from ade import ADE_CLASSES

# Set device to cuda if available, else cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model name
model_name = 'shi-labs/oneformer_ade20k_swin_large'

# Define the path of the image to be segmented
path = os.getcwd().rsplit('/', 1)[0] + "/SegFormer/sample_imgs/FB_IMG_1694996981866.jpg"

# Read the image
read_image = cv2.imread(path)

# The Auto API loads a OneFormerProcessor for us, based on the checkpoint. It internally consists of an image processor
# (for the image modality) and a tokenizer (for the text modality). OneFormer is actually a multimodal model, since it
# incorporates both images and text to solve image segmentation.
processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)

# Start the timer
t0 = time.time()

# Prepare the image for the model
semantic_inputs = processor(images=read_image, task_inputs=["semantic"], return_tensors="pt").to(device)

# for k, v in panoptic_inputs.items():
#     print(k, v.shape)
#
# # We can decode the task inputs back to text:
# print(processor.tokenizer.batch_decode(panoptic_inputs.task_inputs))

# Load the model
model = AutoModelForUniversalSegmentation.from_pretrained(model_name, ).to(device)

# Forward pass through the model
outputs = model(**semantic_inputs)

# Postprocess the raw outputs and visualize the predictions
panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(160, 213)])[0]

# Stop the timer
t1 = time.time()

# Get the classes from the model configuration
classes = model.config.id2label

# Print the keys of the panoptic segmentation
print(panoptic_segmentation.keys())

# Calculate and print the time taken
time_taken = t1 - t0
print(f'Time taken: {time_taken}')


def draw_panoptic_segmentation(segmentation, segments_info):
    """
    Function to draw the panoptic segmentation.

    Parameters:
    segmentation (Tensor): The segmentation tensor.
    segments_info (list): List of segment information.

    Returns:
    None
    """
    # Get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(segmentation.detach().cpu())

    # Initialize a counter for instances
    instances_counter = defaultdict(int)

    # Initialize a list for handles
    handles = []

    # For each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))

    # Add a legend to the axes
    ax.legend(handles=handles)

    # Display the figure
    plt.show()

    # Save the figure
    plt.savefig('cats_panoptic.png')


# Call the function to draw the panoptic segmentation
draw_panoptic_segmentation(**panoptic_segmentation)

# Get the segmentation and segments_info from the panoptic segmentation
segmentation = panoptic_segmentation['segmentation']
segments_info = panoptic_segmentation['segments_info']

# Convert the segmentation to float32, add two dimensions, detach it and move it to cpu
segmentation = segmentation.to(torch.float32).unsqueeze(0).unsqueeze(0).detach().cpu()

# Interpolate the segmentation to the size of the read image
seg = F.interpolate(segmentation, size=read_image.shape[:2], mode='bicubic').squeeze(0).squeeze(0)

# Clone the segmentation and convert it to numpy
result = seg.clone().numpy()

# Initialize a counter for instances
instances_counter = defaultdict(list[str, int])

# Get the segment ids, segment label ids and segment labels
segment_ids = [segment['id'] for segment in segments_info]
segment_label_ids = [segment['label_id'] for segment in segments_info]
segment_labels = [classes[segment_label_id] for segment_label_id in segment_label_ids]

# Copy the result to mask
mask = result.copy()

# Set segmented pixels to 255
mask[mask == segment_ids[2]] = 255

# Set remaining pixels to 0
mask[mask != 255] = 0

# Make mask consistent by dilating and eroding it
kernel = np.ones((9, 9), np.uint8)
mask = cv2.dilate(np.array(mask), kernel, iterations=2)

plt.imshow(mask)
plt.show()

# Alternate
mask = Image.fromarray(mask.astype(np.uint8))
mask.show()
mask.save("mask.png")
