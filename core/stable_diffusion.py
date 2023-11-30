from diffusers import DiffusionPipeline, LCMScheduler, ConsistencyDecoderVAE, AutoencoderTiny, AutoPipelineForText2Image 
import torch

class SD_Model():

    def __init__(self, device, img_size):
        # model_id = "stabilityai/stable-diffusion-xl-base-1.0" 
        # self.model = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        # vae = ConsistencyDecoderVAE.from_pretrained( "openai/consistency-decoder" , torch_dtype=torch.float16)
        self.model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        )

        self.model.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
        self.model.to(device)

        self.img_size = img_size 
        self.generator = torch.Generator(device=device).manual_seed(0)
    
    def enhancer_model(self):

        # add FreeU
        self.model.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)

        # add LCM
        self.model.scheduler = LCMScheduler.from_config(self.model.scheduler.config)
        self.model.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")

    
    def gen(self, prompt, negative_prompt="", steps = 5):
        image = self.model(
            prompt = prompt,
            # strength=0.5,
            negative_prompt = negative_prompt,
            num_inference_steps=steps,
            width = self.img_size[0],
            height = self.img_size[1],
            generator = self.generator,
            guidance_scale = 0,
        ).images[0]
        return image
