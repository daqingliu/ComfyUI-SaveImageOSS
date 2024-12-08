from PIL import Image
from io import BytesIO
import time
import requests
import numpy as np
import comfy.utils
from PIL.PngImagePlugin import PngInfo


class SaveImageOSS:
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
    CATEGORY = "api/image"
    TITLE = "Save Image (OSS)"

    def save_images(self, images, urls):
        start_time = time.time()
        urls = urls.split('\n')
        assert len(urls) == len(images)

        for url, image in zip(urls, images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            meta = PngInfo()
            buffer = BytesIO()
            img.save(buffer, "png", pnginfo=meta, compress_level=0)
            buffer.seek(0)

            with requests.put(url, data=buffer.read()) as r:
                r.raise_for_status()

        return { "ui": {"upload_time": [time.time()-start_time]} }
