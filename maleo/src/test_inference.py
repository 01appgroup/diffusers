
import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, EulerDiscreteScheduler, UNet2DConditionModel

output_path = "/pfs/sshare/app/xujianbo/diffusers/output"

models_conf = {
    "sdxl-he1k": {
        'base': "/pfs/sshare/app/zhangsan/models/stable-diffusion-xl-base-1.0/",
        "ckpt": os.path.join(output_path, "sdxl-he1k"),
    },
    "sdxl-he1k-ema": {
        'base': "/pfs/sshare/app/zhangsan/models/stable-diffusion-xl-base-1.0/",
        "ckpt": os.path.join(output_path, "sdxl-he1k-ema"),
    }
}


prompt = [
    "a picture of a cat in a bucket",
    "a yellow wall with 'Hello, world.' written on it",
    "a painting of the Mona Lisa with New York City behind her"
]


def gen_images(prompt, pipe, size, count):
    pipe.to("cuda")
    seed = torch.Generator("cuda").manual_seed(42)
    images = pipe(prompt=prompt,
                  width=size,
                  height=size,
                  guidance_scale=7, generator=seed,
                  num_images_per_prompt=count).images
    return images


def save_images(name, images_list, size):
    header_size = 64
    merged = Image.new(images_list[0]['imgs'][0].mode,
                       (size*len(images_list), header_size + size*len(images_list[0]['imgs'])))
    draw = ImageDraw.Draw(merged)
    font_size = 48
    font = ImageFont.truetype(os.path.join(output_path, 'DejaVuSans.ttf'), font_size)

    for col, item in enumerate(images_list):
        # bbox = draw.textbbox(item['name'], align='center')
        draw.text((size*col, 2), item['name'], font=font)
        for row, img in enumerate(item['imgs']):
            merged.paste(img, (size*col, header_size + size*row))

    fn = f"{name}_{len(images_list)}.jpg"
    merged.save(os.path.join(output_path, fn))
    print(f"save to {fn} OK")


def main(conf_name: str, prompt, image_size=1024, img_per_prompt=1):
    conf = models_conf[conf_name]
    # load origin:
    pipe: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
        conf['base'], torch_dtype=torch.float16, use_safetensors=True,
        variant="fp16")

    # # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)    
    img_list = [
        {
            "name": "base",
            "imgs": gen_images(prompt, pipe, size=image_size, count=img_per_prompt)
        }
    ]

    pipe_ckpt: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
        conf['ckpt'], torch_dtype=torch.float16, use_safetensors=True,
        variant="fp16")

    # load checkpoint
    for ckpt_name in os.listdir(conf['ckpt']):
        if ckpt_name.startswith("checkpoint-"):
            pipe_ckpt.unet = UNet2DConditionModel.from_pretrained(os.path.join(conf['ckpt'], ckpt_name), 
                subfolder="unet", torch_dtype=torch.float16, use_safetensors=True)
            img_list.append(
                {
                    "name": ckpt_name,
                    "imgs": gen_images(prompt, pipe_ckpt, size=image_size, count=img_per_prompt)
                }
            )

    save_images(f"{conf_name}-{image_size}_{img_per_prompt}", img_list, size=image_size)


if __name__ == "__main__":
    # conf_name = "sdxl-he1k"
    conf_name = "sdxl-he1k-ema"
    main(conf_name, prompt=prompt, image_size=1024, img_per_prompt=1)
