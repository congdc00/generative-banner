from diffusers import StableDiffusionLatentUpscalePipeline
import torch
from PIL import Image
class Upscaler():
    model = None
    
    @staticmethod
    def init_model():
        model_id = "stabilityai/sd-x2-latent-upscaler"
        Upscaler.model = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        Upscaler.model.to("cuda")

    @staticmethod
    def upscale(low_res_img, generator):
        promt = "8k, hd"
        if Upscaler.model == None:
            Upscaler.init_model()

        upscaled_image = Upscaler.model(
            prompt=promt,
            image=low_res_img,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
        ).images[0]
        
        return upscaled_image

if __name__ == "__main__":
    generator = torch.Generator(device="cuda").manual_seed(0)

    image = Image.open("test.jpg")
    image = Upscaler.upscale(image, generator)
    image.save("test_02.png")    



