import torch
from diffusers import UniPCMultistepScheduler
from utils.pipeline_t2i import StableDiffusionPipeline
from ssr_encoder import SSR_encoder
import gradio as gr
from PIL import Image

base_model_path = "sd_model_v1-5". # your sd15 path
image_encoder_path = "models/image_encoder"
base_ssr = "./models/ssr_model"
ssr_ckpt = [base_ssr+"/pytorch_model.bin",
            base_ssr+"/pytorch_model_1.bin"]

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    safety_checker=None,
    torch_dtype=torch.float32).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

sc_model = SSR_encoder(pipe.unet, image_encoder_path, "cuda", dtype=torch.float32)
sc_model.get_pipe(pipe)
sc_model.load_SSR(ssr_ckpt[0], ssr_ckpt[1])

# Define your ML model or function here
def model_call(pil_img, subject, prompts, negative_prompt, scale):
    # Your ML logic goes here
    pil_img = Image.fromarray(pil_img.astype('uint8'), 'RGB')
    images = sc_model.generate(
        pil_image=pil_img,
        concept=subject,
        uncond_concept="",
        prompt=prompts,
        negative_prompt=negative_prompt,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        scale=scale,
        num_inference_steps=30,
        height=512,
        width=512,
    )[0]
    return images

# Create a Gradio interface
iface = gr.Interface(
    fn=lambda image, text1, text2, text3, num, : model_call(image, text1, text2, text3, num),
    inputs=[gr.Image(), \
        gr.Textbox(label="subject query"), \
        gr.Textbox(label="prompts"), \
        gr.Textbox(label="neg prompts"), \
        gr.inputs.Slider(minimum=0, maximum=2, default=0.65, label="subject_scale")],
    outputs=gr.Image(),
    title="SSR Demo",
    description="Upload an image and enter text inputs to see the model output."
)
# Launch the Gradio interface
iface.launch(server_name='0.0.0.0')
