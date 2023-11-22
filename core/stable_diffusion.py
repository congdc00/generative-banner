from diffusers import DiffusionPipeline, LCMScheduler    
import torch

class SD_Model():

    def __init__(self, device, img_size):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0" 
        self.model = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        self.img_size = img_size 
        self.generator = torch.Generator(device=device).manual_seed(0)
    
    def enhancer_model(self):

        # add FreeU
        self.model.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)

        # add LCM
        self.model.scheduler = LCMScheduler.from_config(self.model.scheduler.config)
        self.model.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")

    
    def gen(self, prompt, negative_promt="", steps = 4):
        image = self.model(
            prompt = prompt,
            # strength=0.5,
            # negative_prompt = negative_prompt,
            num_inference_steps=steps,
            width = self.img_size[0],
            height = self.img_size[1],
            generator = self.generator,
            guidance_scale = 1.0,
        ).images[0]
        return image
