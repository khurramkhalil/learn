# Import the necessary libraries
import torch
from diffusers import StableDiffusionPipeline

# Check if CUDA is available, if it is, use it, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model name
model_name = 'runwayml/stable-diffusion-v1-5'

# Get the free memory in GB on the GPU
free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)

# Define the maximum memory to be used by the model, which is the free memory minus 2GB
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

# Get the number of GPUs available
n_gpus = torch.cuda.device_count()

# Create a dictionary where the keys are the GPU indices and the values are the maximum memory
max_memory = {i: max_memory for i in range(n_gpus)}

# Load the pretrained model from the specified path, set the data type to float16, and ensure that the model is loaded locally
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True).to(device)

# Define the prompts for the model
prompts = "a photo of an astronaut riding a horse on mars, photorealistic, 4k, highly detailed"

prompt = "Inpaint a modern, elegant queen-sized bed with white bedding and plush pillows, centered in the room,"\
          "under soft, warm lightening, creating cozy and inviting atmosphere."

# Generate an image based on the prompt
image = pipe(prompt).images[0]

# Display the generated image
image.show()

# Save the generated image to a file
image.save("astronaut_rides_horse.png")
