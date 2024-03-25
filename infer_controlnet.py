import os.path
import torch
from diffusers import UniPCMultistepScheduler, ControlNetModel
from utils.pipeline_controlnet import StableDiffusionControlNetPipeline
from ssr_encoder import SSR_encoder
from diffusers.utils import load_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# model_paras
controlnet_path = "control_v11f1p_sd15_depth"  # your controlnet path
base_model_path = "sd_model_v1-5"  # your sd15 path
image_encoder_path = "models/image_encoder"
base_ssr = "./models/ssr_model"
ssr_ckpt = [base_ssr+"/pytorch_model.bin",
            base_ssr+"/pytorch_model_1.bin"]

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
ssr_model = SSR_encoder(pipe.unet, image_encoder_path, "cuda", dtype=torch.float16)
ssr_model.get_pipe(pipe)
ssr_model.load_SSR(ssr_ckpt[0], ssr_ckpt[1])

if __name__ == '__main__':

    # paras
    subject = "cat"
    scale = 0.7
    prompt = "A cat in the forest"
    out_dir = "./results"
    img_path = "./test_img/controlnet/1.jpg"
    control_img = "./test_img/controlnet/dog_depth.png"
    negative_prompt = "bad quality"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pil_img = load_image(img_path)
    control = load_image(control_img)
    images = ssr_model.generate_control(
        pil_image=pil_img,
        control_img=control,
        concept=subject,
        uncond_concept="",
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_samples=1,
        seed=None,
        guidance_scale=5,
        scale=scale,
        num_inference_steps=30,
        height=512,
        width=512,
    )[0]
    images.save(os.path.join(out_dir, "control.jpg"))