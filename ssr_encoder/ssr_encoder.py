from typing import List
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
if is_torch2_available():
    from .attention_processor import SSRAttnProcessor2_0 as SSRAttnProcessor, AttnProcessor2_0 as AttnProcessor, AlignerAttnProcessor
else:
    from .attention_processor import SSRAttnProcessor, AttnProcessor, AlignerAttnProcessor1_0 as AlignerAttnProcessor
from diffusers.models.attention_processor import Attention
from .attention_processor import get_attention_scores_mask, AlignerAttnProcessor_mask

class Aligner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aligner = Attention(
            query_dim=768,
            cross_attention_dim=1024,
            heads=8,
            dim_head=64,
            dropout=0.,
        )
        self.norm = torch.nn.LayerNorm(768)
        self.aligner.to_v = None
        self.aligner.to_out = None  # for paral train

    def forward(self, text_embeds, image_embeds, mask=None, mask_weight=None):
        if mask is None:
            subject_embeds = self.aligner(text_embeds, encoder_hidden_states=image_embeds)
        else:
            subject_embeds = self.aligner(text_embeds, encoder_hidden_states=image_embeds, mask=mask, mask_weight=mask_weight)
        subject_embeds = self.norm(subject_embeds)
        return subject_embeds

class SSR_encoder(torch.nn.Module):
    """SSR-encoder"""

    def __init__(self, unet, image_encoder_path, device="cuda", dtype=torch.float32, is_animate=False, is_mask=False):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # load image encoder
        self.image_encoder = CLIPVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
        self.clip_image_processor = CLIPImageProcessor()

        # load SSR layers
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if is_animate:
                if name.endswith("attn1.processor") or name.endswith(
                        "motion_modules.0.transformer_blocks.0.attn1.processor") \
                        or name.endswith("motion_modules.0.transformer_blocks.0.attn2.processor") \
                        or name.endswith("motion_modules.0.transformer_blocks.1.attn1.processor") \
                        or name.endswith("motion_modules.0.transformer_blocks.1.attn2.processor") \
                        or name.endswith("motion_modules.0.transformer_blocks.2.attn1.processor") \
                        or name.endswith("motion_modules.1.transformer_blocks.0.attn1.processor") \
                        or name.endswith("motion_modules.1.transformer_blocks.0.attn2.processor") \
                        or name.endswith("motion_modules.2.transformer_blocks.0.attn1.processor") \
                        or name.endswith("motion_modules.2.transformer_blocks.0.attn2.processor") \
                        or name.endswith("motion_modules.0.transformer_blocks.2.attn2.processor"):
                    cross_attention_dim = None
                else:
                    cross_attention_dim = unet.config.cross_attention_dim
            else:
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = SSRAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                    scale=1).to(self.device, dtype=self.dtype)
        unet.set_attn_processor(attn_procs)

        if is_animate:
            layer_modules = torch.nn.ModuleList([])
            for k, v in unet.attn_processors.items():
                if k.endswith('attentions.0.transformer_blocks.0.attn2.processor') \
                        or k.endswith('attentions.1.transformer_blocks.0.attn2.processor') \
                        or k.endswith('attentions.2.transformer_blocks.0.attn2.processor'):
                    layer_modules.append(v)
        else:
            layer_modules = torch.nn.ModuleList(unet.attn_processors.values())

        # load SSR
        self.SSR_aligners = Aligner()

        if is_mask:
            Aligner_procs = AlignerAttnProcessor_mask(query_dim=768, inner_dim=512, cross_attention_dim=1024)
            self.SSR_aligners.aligner.get_attention_scores_mask = get_attention_scores_mask.__get__(self.SSR_aligners.aligner)
            self.SSR_aligners.aligner.set_processor(Aligner_procs)
        else:
            Aligner_procs = AlignerAttnProcessor(query_dim=768, inner_dim=512, cross_attention_dim=1024)
            self.SSR_aligners.aligner.set_processor(Aligner_procs)

        self.SSR_layers = layer_modules
        self.SSR_aligners.to(self.device, dtype=self.dtype)
        self.SSR_layers.to(self.device, dtype=self.dtype)

    def load_SSR(self, aligner_path, layers_path):
        state_dict1 = torch.load(aligner_path, map_location=self.device)
        self.SSR_aligners.load_state_dict(state_dict1)
        state_dict0 = torch.load(layers_path, map_location=self.device)
        self.SSR_layers.load_state_dict(state_dict0)

    def animate_load_SSR(self, aligner_path, layers_path):
        state_dict1 = torch.load(aligner_path, map_location=self.device)
        self.SSR_aligners.load_state_dict(state_dict1)
        state_dict0 = torch.load(layers_path, map_location=self.device)
        new_odict = {}
        for k, v in state_dict0.items():
            new_odict[str((int(k.split(".")[0]) - 1) // 2) + "." + k.split(".")[1] + "." + k.split(".")[2]] = v
        self.SSR_layers.load_state_dict(new_odict)

#     def load_org_ssr(self, aligner_path, layers_path):
#         state_dict1 = torch.load(aligner_path, map_location=self.device)
#         new_odict = {}
#         for k, v in state_dict1.items():
#             if 'attn' in k:
#                 words = k.split(".")
#                 words[0] = "aligner"
#                 new_k = ".".join(words)
#                 new_odict[new_k] = v
#             else:
#                 new_odict[k] = v
#         self.SSR_aligners.load_state_dict(new_odict, strict=False)

#         state_dict0 = torch.load(layers_path, map_location=self.device)
#         new_odict = {}
#         for k, v in state_dict0.items():
#             if "to_k_SC" in k:
#                 words = k.split(".")
#                 words[1] = "to_k_SSR"
#                 new_k = ".".join(words)
#                 new_odict[new_k] = v
#             else:
#                 words = k.split(".")
#                 words[1] = "to_v_SSR"
#                 new_k = ".".join(words)
#                 new_odict[new_k] = v
#         self.SSR_layers.load_state_dict(new_odict, strict=False)

    def forward(self, encoder_hidden_states, image_embeds):
        subject_embeds = self.SSR_aligners(text_embeds=encoder_hidden_states, image_embeds=image_embeds)
        return subject_embeds

    def get_pipe(self, pipe):
        self.pipe = pipe

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, SSRAttnProcessor):
                attn_processor.scale = scale

    def generate(
            self,
            pil_image,
            concept,
            uncond_concept=" ",
            prompt=" ",
            negative_prompt=" ",
            scale=1.0,
            num_samples=1,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            height=512,
            width=512,
    ):
        self.set_scale(scale)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image, num_samples, concept, uncond_concept
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images
        return images

    def generate_control(
            self,
            pil_image,
            control_img,
            concept,
            uncond_concept=" ",
            prompt=" ",
            negative_prompt=" ",
            scale=1.0,
            num_samples=1,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            height=512,
            width=512,
    ):
        self.set_scale(scale)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image, num_samples, concept, uncond_concept
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.pipe(
            height=height,
            width=width,
            image=control_img,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=1.,
        ).images
        return images

    def generate_animate(
            self,
            pil_image,
            concept,
            num_frames=16,
            uncond_concept=" ",
            prompt=" ",
            negative_prompt=" ",
            scale=1.0,
            num_samples=1,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            height=512,
            width=512,
    ):
        self.set_scale(scale)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image, num_samples, concept, uncond_concept
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        output = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
        )
        frames = output.frames[0]
        return frames

    def generate_multi(
            self,
            pil_image_list,
            subject_list,
            uncond_concept=[],
            prompt=[],
            negative_prompt=[],
            scale=[],
            num_samples=1,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            height=512,
            width=512,
    ):
        self.num = len(pil_image_list)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image_list, num_samples, subject_list, uncond_concept
        )
        image_prompt_embeds = image_prompt_embeds.view(1, -1, 768)
        cache = []
        for i in range(len(scale)):
            image_prompt_embed_new = scale[i] * image_prompt_embeds[:, int(77 * self.num * i * 6): \
                                            int(77 * self.num * (i+1) * 6), :]
            image_prompt_embed = image_prompt_embed_new
            cache.append(image_prompt_embed)
        image_prompt_embeds = torch.cat(cache, dim=1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(1, -1, 768)

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images
        return images

    def generate_mask(
            self,
            mask,
            pil_image,
            subject,
            prompt=" ",
            negative_prompt=" ",
            scale=1.0,
            num_samples=1,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            height=640,
            width=640,
            mask_weight=5,
    ):
        self.set_scale(scale)
        self.mask_trans = transforms.ToTensor()

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_mask(
            pil_image, num_samples, subject, "", mask, mask_weight
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images
        return images

    @torch.inference_mode()
    def get_image_embeds(self, pil_image, num_samples, text, uncond_text):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = []
        for pil in pil_image:
            tensor_image = self.clip_image_processor(images=pil, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
            clip_image.append(tensor_image)
        clip_image = torch.cat(clip_image, dim=0)

        # text
        prompt_embeds = self.pipe._encode_prompt(
            text, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
            negative_prompt=uncond_text)
        negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)

        # cond
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True)['hidden_states'][4::4]
        clip_image_embeds = torch.cat(clip_image_embeds, dim=1)
        image_prompt_embeds = self.SSR_aligners(
            prompt_embeds_,
            clip_image_embeds
        )

        uncond_clip_image_embeds = \
        self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True)['hidden_states'][4::4]
        uncond_clip_image_embeds = torch.cat(uncond_clip_image_embeds, dim=1)
        uncond_image_prompt_embeds = self.SSR_aligners(
            negative_prompt_embeds_,
            uncond_clip_image_embeds
        )

        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_mask(self, pil_image, num_samples, text, uncond_text, masks, mask_weight):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = []
        for pil in pil_image:
            tensor_image = self.clip_image_processor(images=pil, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
            clip_image.append(tensor_image)
        clip_image = torch.cat(clip_image, dim=0)

        if isinstance(masks, Image.Image):
            masks = [masks]
        mask_image = []
        for mask in masks:
            tensor_image = self.mask_trans(mask).unsqueeze(0)
            tensor_image = torch.where(tensor_image > 0.5, torch.ones_like(tensor_image),
                                       torch.zeros_like(tensor_image))
            mask_image.append(tensor_image)
        mask_image = torch.cat(mask_image, dim=0)

        # text
        prompt_embeds = self.pipe._encode_prompt(
            text, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
            negative_prompt=uncond_text)
        negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)

        # cond
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True)['hidden_states'][4::4]
        clip_image_embeds = torch.cat(clip_image_embeds, dim=1)
        image_prompt_embeds = self.SSR_aligners(
            prompt_embeds_,
            clip_image_embeds,
            mask=mask_image,
            mask_weight=mask_weight
        )

        uncond_clip_image_embeds = \
            self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True)['hidden_states'][4::4]
        uncond_clip_image_embeds = torch.cat(uncond_clip_image_embeds, dim=1)
        uncond_image_prompt_embeds = self.SSR_aligners(
            negative_prompt_embeds_,
            uncond_clip_image_embeds,
            mask=mask_image,
            mask_weight=mask_weight
        )

        return image_prompt_embeds, uncond_image_prompt_embeds