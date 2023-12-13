# Import necessary libraries
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInpaintPipeline

# Define the paths to the image and mask files
path_image = os.getcwd().rsplit('/', 1)[0] + "/SegFormer/sample_imgs/FB_IMG_1694996981866.jpg"
path_mask = os.getcwd().rsplit('/', 1)[0] + "/SegFormer/sample_imgs/mask.png"

# Function to resize an image to the nearest multiple of 64
def resize_to_nearest_64(image_path):
    # Open the image
    with Image.open(image_path) as img:
        # Calculate the nearest multiple of 64 for width and height
        # new_width = round(img.width / 64) * 64
        # new_height = round(img.height / 64) * 64
        # For now, we are resizing to 512x512
        new_width = 512
        new_height = 512

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.NEAREST)
        return resized_img

# Resize the image and mask
image = resize_to_nearest_64(path_image)
mask_image = resize_to_nearest_64(path_mask)

# Create a kernel for dilation operation
kernel = np.ones((9, 9), np.uint8)
# Dilate the mask image
result = cv2.dilate(np.array(mask_image), kernel, iterations=2)
# Convert the result back to a PIL image
new_mask = Image.fromarray(result.astype(np.uint8))

# Check if CUDA is available, else use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Define the model name
model_name = 'runwayml/stable-diffusion-inpainting'

# Start the timer
t0 = time.time()
# Load the model
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, revision="fp16", torch_dtype=torch.float16,
                                                      local_files_only=True, ).to(device)

# Define the prompts for the inpainting task
prompt1 = ("A big black painting of a horse (1.0), flaunting hairs, hanging on the wall. "
           "beautiful, realistic (1.0), seamless integration (1.0)"
           ", 4k, photo realistic")

prompt2 = ("Inpaint a highly detailed and photorealistic drawing of a horse on th wall of this room. The horse "
           "should be fully sized, fitting naturally within the room's dimensions. It should be depicted in calm, "
           "standing pose, with its head slightly turned towards left, giving a sense of life and presence. The"
           "horse coat's should be a rich chestnut color, with a glossy sheen reflecting the room's lightening,"
           "adding depth and realism.")

prompt = ("Inpaint a modern, elegant queen-sized bed with white bedding and plush pillows, centered in the room,"
          "under soft, warm lightening, creating cozy and inviting atmosphere.")

# Define the negative prompt
neg_prompt = (
    "window, door, fireplace, text, word, cropped, low quality, normal quality, username, watermark, signature, "
    "blurry, soft, soft line, curved line, sketch, ugly, logo, pixelated, low-res")

# Perform the inpainting task
inpaint_image = pipe(prompt=prompt,
                     image=image,
                     mask_image=new_mask,
                     num_images_per_prompt=4,
                     guidance_scale=7.5,
                     negative_prompt=neg_prompt,
                     ).images

# Stop the timer and calculate the time taken
t1 = time.time()
time_taken = t1 - t0
print(f'Time taken: {time_taken}')

# Create a new canvas to paste the inpainted images
im_width, im_height = inpaint_image[0].size
canvas = Image.new('RGB', (im_width * len(inpaint_image), im_height))
# Paste the inpainted images onto the canvas
for i, img in enumerate(inpaint_image):
    canvas.paste(img, (i * im_width, 0))

# Show the final result
canvas.show()
# Save the final result
canvas.save("./big_black_painting.png")
