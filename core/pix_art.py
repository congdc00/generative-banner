from diffusers import PixArtAlphaPipeline
import torch 
class PixArt_Model():

    def __init__(self, device, img_size):
        self.model = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16).to(device)
        self.model.enable_model_cpu_offload()
    
    def gen(self, prompt, negative_promt="", steps = 65):
        image = self.model(prompt = prompt).images[0]
        return image
