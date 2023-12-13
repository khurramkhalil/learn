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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'shi-labs/oneformer_ade20k_swin_large'

path = os.getcwd().rsplit('/', 1)[0] + "/SegFormer/sample_imgs/FB_IMG_1694996981866.jpg"
read_image = cv2.imread(path)

# The Auto API loads a OneFormerProcessor for us, based on the checkpoint. It internally consists of an image processor
# (for the image modality) and a tokenizer (for the text modality). OneFormer is actually a multimodal model, since it
# incorporates both images and text to solve image segmentation.
processor = AutoProcessor.from_pretrained(model_name)

t0 = time.time()
# prepare image for the model
panoptic_inputs = processor(images=read_image, task_inputs=["panoptic"], return_tensors="pt").to(device)

# for k, v in panoptic_inputs.items():
#     print(k, v.shape)
#
# # We can decode the task inputs back to text:
# print(processor.tokenizer.batch_decode(panoptic_inputs.task_inputs))

model = AutoModelForUniversalSegmentation.from_pretrained(model_name, ).to(device)

# Forward pass
outputs = model(**panoptic_inputs)

# Postprocess the raw outputs and visualize the predictions.
panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(160, 213)])[0]
t1 = time.time()

classes = model.config.id2label
print(panoptic_segmentation.keys())

time_taken = t1 - t0
print(f'Time taken: {time_taken}')


def draw_panoptic_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation.detach().cpu())
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles)
    plt.show()
    plt.savefig('cats_panoptic.png')


# draw_panoptic_segmentation(**panoptic_segmentation)

segmentation = panoptic_segmentation['segmentation']
segments_info = panoptic_segmentation['segments_info']

segmentation = segmentation.to(torch.float32).unsqueeze(0).unsqueeze(0).detach().cpu()
seg = F.interpolate(segmentation, size=read_image.shape[:2], mode='bicubic').squeeze(0).squeeze(0)
result = seg.clone().numpy()
instances_counter = defaultdict(list[str, int])

segment_ids = [segment['id'] for segment in segments_info]
segment_label_ids = [segment['label_id'] for segment in segments_info]
segment_labels = [classes[segment_label_id] for segment_label_id in segment_label_ids]

mask = result.copy()
mask[mask == segment_ids[0]] = 255

plt.imshow(mask)
plt.show()

# Alternate
mask = Image.fromarray(mask.astype(np.uint8))
mask.show()
