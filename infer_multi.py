import os.path
import torch
from diffusers import UniPCMultistepScheduler
from utils.pipeline_t2i import StableDiffusionPipeline
from ssr_encoder import SSR_encoder
from diffusers.utils import load_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

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
    num = 3  # num of images
    subjects = ["dog", "man", "flowers, mountain"]
    scales = [0.8, 1.2, 1.]  # need to balance scales
    prompt = "A man holding a dog"
    out_dir = "./results"
    img_path = "./test_img/multi"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pil_img_ls = []
    for idx, name in enumerate(sorted(os.listdir(img_path))):
        if not is_image_file(name):
            continue
        print(name)
        pil_img_ls.append(load_image(os.path.join(img_path, name)))

    images = ssr_model.generate_multi(
        pil_image_list=pil_img_ls,
        subject_list=subjects,
        uncond_concept=[""]*num,
        prompt=prompt,
        negative_prompt="",
        num_samples=1,
        seed=None,
        guidance_scale=5,
        scale=scales,
        num_inference_steps=30,
        height=512,
        width=512,
    )[0]
    images.save(os.path.join(out_dir, "multi.jpg"))