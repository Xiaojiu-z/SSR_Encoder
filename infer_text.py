import os.path
import torch
from diffusers import UniPCMultistepScheduler
from utils.pipeline_t2i import StableDiffusionPipeline
from ssr_encoder import SSR_encoder
from diffusers.utils import load_image

# Initialize the model
base_model_path = "sd_model_v1-5"
image_encoder_path = "models/image_encoder"
base_ssr = "./models/ssr_model"
ssr_ckpt = [base_ssr+"/pytorch_model.bin",
            base_ssr+"/pytorch_model_1.bin"]

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    safety_checker=None,
    torch_dtype=torch.float32).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
ssr_model = SSR_encoder(pipe.unet, image_encoder_path, "cuda", dtype=torch.float32)
ssr_model.get_pipe(pipe)
ssr_model.load_SSR(ssr_ckpt[0], ssr_ckpt[1])

if __name__ == '__main__':

    # infer paras
    img_path = "./test_img/t2i/3.jpg"
    out_path = "./results"
    subject = "flower"
    scale = 0.65  # The recommended parameters are 0.5-0.8, Default value is 0.65
    prompts = "A girl holding flowers"
    negative_prompt = "bad quality"

    pil_img = load_image(img_path)
    images = ssr_model.generate(
        pil_image=pil_img,
        concept=subject,
        uncond_concept="",
        prompt=prompts,
        negative_prompt=negative_prompt,
        num_samples=1,
        seed=None,
        guidance_scale=5,
        scale=scale,
        num_inference_steps=30,
        height=512,
        width=512,
    )[0]
    images.save(os.path.join(out_path, "t2i.jpg"))