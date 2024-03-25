import os.path
import torch
from diffusers import MotionAdapter, AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers.utils import export_to_gif
from diffusers.utils import load_image
from ssr_encoder import SSR_encoder
from utils.pipeline_animatediff import AnimateDiffPipeline

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
# model_paras
base_model_path = "sd_model_v1-5"
image_encoder_path = "models/image_encoder"
adapter = MotionAdapter.from_pretrained("motion_v1-5-2")  # animate_diff model path
vae_path = "vae_ft"  # recommended vae from animate_diff
base_ssr = "./models/ssr_model"
ssr_ckpt = [base_ssr+"/pytorch_model.bin",
            base_ssr+"/pytorch_model_1.bin"]

# load models
pipe = AnimateDiffPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    torch_dtype=torch.float16).to("cuda")
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to("cuda")
pipe.vae = vae
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
ssr_model = SSR_encoder(pipe.unet, image_encoder_path, "cuda", dtype=torch.float16, is_animate=True)
ssr_model.get_pipe(pipe)
ssr_model.animate_load_SSR(ssr_ckpt[0], ssr_ckpt[1])

if __name__ == '__main__':

    # paras
    subject = "girl"
    scale = 0.6
    prompt = "A girl is dancing, 4k"
    out_dir = "./results"
    img_path = "./test_img/animate/1.jpeg"
    neg = "bad quality, blur"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    pil_img = load_image(img_path)
    frames = ssr_model.generate_animate(
        num_frames=16,
        pil_image=pil_img,
        concept=subject,
        uncond_concept="",
        prompt=prompt,
        negative_prompt=neg,
        num_samples=1,
        seed=None,
        guidance_scale=5,
        scale=scale,
        num_inference_steps=30,
        height=512,
        width=512,
    )
    export_to_gif(frames, os.path.join(out_dir, "animation.gif"))