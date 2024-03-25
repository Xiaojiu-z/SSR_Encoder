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
ssr_model = SSR_encoder(pipe.unet, image_encoder_path, "cuda", dtype=torch.float32, is_mask=True)
ssr_model.get_pipe(pipe)
ssr_model.load_SSR(ssr_ckpt[0], ssr_ckpt[1])

if __name__ == '__main__':

    # infer paras
    prompt = "a girl in a snowy day"
    out_path = "./results"
    img_path = "./test_img/mask/4.jpg"
    mask_path = "./test_img/mask/girl.png"
    caption = "A man wearing a black suit and a girl wearing a red dress"  # maybe blip2

    pil_img = load_image(img_path)
    pil_img = pil_img.resize((512, 512))
    mask = load_image(mask_path)
    mask = mask.resize((512, 512))

    images = ssr_model.generate_mask(
        mask=mask,  # use mask to further determine subject that the cross-attn focus on
        pil_image=pil_img,
        subject=caption,  # need to use captions to get all subjects
        prompt=prompt,
        negative_prompt="",
        scale=0.5,
        num_samples=1,
        seed=None,
        guidance_scale=5,
        num_inference_steps=25,
        height=512,
        width=512,
        mask_weight=1,  # The effect of mask_weight needs to be adjusted manually, so I recommend using text
    )[0]

    images.save(os.path.join(out_path, "mask_girl.jpg"))