import os
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInpaintPipeline

path_image = os.getcwd().rsplit('/', 1)[0] + "/SegFormer/sample_imgs/FB_IMG_1694996981866.jpg"
path_mask = os.getcwd().rsplit('/', 1)[0] + "/SegFormer/sample_imgs/mask.png"


def resize_to_nearest_64(image_path):
    # Open the image
    with Image.open(image_path) as img:
        # Calculate the nearest multiple of 64 for width and height
        new_width = 512
        new_height = 512

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.NEAREST)
        return resized_img


image = resize_to_nearest_64(path_image)
mask_image = resize_to_nearest_64(path_mask)

kernel = np.ones((9, 9), np.uint8)
result = cv2.dilate(np.array(mask_image), kernel, iterations=2)
new_mask = Image.fromarray(result.astype(np.uint8))
# new_mask = ImageOps.invert(new_mask)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'

t0 = time.time()
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to(device)
                                                      # local_files_only=True, ).to(device)
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

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

neg_prompt = (
    "window, door, fireplace, text, word, cropped, low quality, normal quality, username, watermark, signature, "
    "blurry, soft, soft line, curved line, sketch, ugly, logo, pixelated, low-res")

generator = torch.Generator("cuda").manual_seed(92)

# The mask structure is white for inpainting and black for keeping as is, image and mask_image should be PIL images.
inpaint_image = pipe(prompt=prompt,
                     image=image,
                     mask_image=new_mask,
                     num_images_per_prompt=4,
                     # control_strength=0.4,
                     guidance_scale=7.5,
                     # num_inference_step=30,
                     negative_prompt=neg_prompt,
                     generator=generator
                     ).images
t1 = time.time()
time_taken = t1 - t0
print(f'Time taken: {time_taken}')

im_width, im_height = inpaint_image[0].size
canvas = Image.new('RGB', (im_width * len(inpaint_image), im_height))
for i, img in enumerate(inpaint_image):
    canvas.paste(img, (i * im_width, 0))

canvas.show()
canvas.save("./sd-xl.png")
