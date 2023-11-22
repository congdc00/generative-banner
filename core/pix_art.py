from diffusers import PixArtAlphaPipeline
    
class PixArt_model():

    def __init__(self, device):
        self.model = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16).to(device)
        self.model.enable_model_cpu_offload()
    
    def gen(self, promt, negative_promt=""):
        image = self.model(promt = promt).images[0]
        return image
