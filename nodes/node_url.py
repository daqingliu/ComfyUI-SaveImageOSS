from PIL import Image
from io import BytesIO
import time
import requests
import numpy as np
from PIL.PngImagePlugin import PngInfo

import io
import os
import torch
import random
import base64
import folder_paths
from typing import List, Dict, Tuple
from PIL import Image, ImageOps, ImageFilter


class SaveImageURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "urls": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "oss/image"
    TITLE = "Save Image (URL)"

    def save_images(self, images, urls):
        urls = urls.split('\n')
        assert len(urls) == len(images)
        results = list()

        for url, image in zip(urls, images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            meta = PngInfo()
            buffer = BytesIO()
            img.save(buffer, "png", pnginfo=meta, compress_level=0)
            buffer.seek(0)

            with requests.put(url, data=buffer.read()) as r:
                r.raise_for_status()

            results.append(url)

        return { "ui": { "url_list": results } }


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)


def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)


def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def prepare_image_for_preview(image: Image.Image, output_dir: str, prefix=None):
    if prefix is None:
        prefix = "preview_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    # save image to temp folder
    (
        outdir,
        filename,
        counter,
        subfolder,
        _,
    ) = folder_paths.get_save_image_path(prefix, output_dir, image.width, image.height)
    file = f"{filename}_{counter:05}_.png"
    image.save(os.path.join(outdir, file), format="PNG", compress_level=4)

    return {
        "filename": file,
        "subfolder": subfolder,
        "type": "temp",
    }


def load_images_from_url(urls: List[str], keep_alpha_channel=False):
    images: List[Image.Image] = []
    masks: List[Image.Image] = []

    for url in urls:
        if url.startswith("data:image/"):
            i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        elif url.startswith("file://"):
            url = url[7:]
            if not os.path.isfile(url):
                raise Exception(f"File {url} does not exist")

            i = Image.open(url)
        elif url.startswith("http://") or url.startswith("https://"):
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise Exception(response.text)

            i = Image.open(io.BytesIO(response.content))
        elif url.startswith(("/view?", "/api/view?")):
            from urllib.parse import parse_qs

            qs_idx = url.find("?")
            qs = parse_qs(url[qs_idx + 1 :])
            filename = qs.get("name", qs.get("filename", None))
            if filename is None:
                raise Exception(f"Invalid url: {url}")

            filename = filename[0]
            subfolder = qs.get("subfolder", None)
            if subfolder is not None:
                filename = os.path.join(subfolder[0], filename)

            dirtype = qs.get("type", ["input"])
            if dirtype[0] == "input":
                url = os.path.join(folder_paths.get_input_directory(), filename)
            elif dirtype[0] == "output":
                url = os.path.join(folder_paths.get_output_directory(), filename)
            elif dirtype[0] == "temp":
                url = os.path.join(folder_paths.get_temp_directory(), filename)
            else:
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)
        elif url == "":
            continue
        else:
            url = folder_paths.get_annotated_filepath(url)
            if not os.path.isfile(url):
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)

        i = ImageOps.exif_transpose(i)
        has_alpha = "A" in i.getbands()
        mask = None

        if "RGB" not in i.mode:
            i = i.convert("RGBA") if has_alpha else i.convert("RGB")

        if has_alpha:
            mask = i.getchannel("A")

        if not keep_alpha_channel:
            image = i.convert("RGB")
        else:
            image = i

        images.append(image)
        masks.append(mask)

    return (images, masks)


class LoadImageURL:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "image": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "output_mode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "list", "label_off": "batch"},
                ),
                "url": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    OUTPUT_IS_LIST = (True, True, False)
    RETURN_NAMES = ("images", "masks", "has_image")
    CATEGORY = "oss/image"
    FUNCTION = "load_image"

    def load_image(self, image="", keep_alpha_channel=False, output_mode=False, url=""):
        if not image or image == "":
            image = url

        urls = image.strip().split("\n")
        images, masks = load_images_from_url(urls, keep_alpha_channel)
        if len(images) == 0:
            image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images = [tensor2pil(image)]
            masks = [tensor2pil(mask, mode="L")]

        previews = []
        np_images = []
        np_masks = []

        for image, mask in zip(images, masks):
            if mask is not None:
                preview_image = Image.new("RGB", image.size)
                preview_image.paste(image, (0, 0))
                preview_image.putalpha(mask)
            else:
                preview_image = image

            previews.append(prepare_image_for_preview(preview_image, self.output_dir, self.filename_prefix))

            image = pil2tensor(image)
            if mask:
                mask = np.array(mask).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            np_images.append(image)
            np_masks.append(mask.unsqueeze(0))

        if output_mode:
            result = (np_images, np_masks, True)
        else:
            has_size_mismatch = False
            if len(np_images) > 1:
                for image in np_images[1:]:
                    if image.shape[1] != np_images[0].shape[1] or image.shape[2] != np_images[0].shape[2]:
                        has_size_mismatch = True
                        break

            if has_size_mismatch:
                raise Exception("To output as batch, images must have the same size. Use list output mode instead.")

            result = ([torch.cat(np_images)], [torch.cat(np_masks)], True)

        return {"ui": {"images": previews}, "result": result}
