from PIL import Image
from io import BytesIO
import time
import requests
import numpy as np
from PIL.PngImagePlugin import PngInfo


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
    CATEGORY = "api/image"
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

            results.append({
                "filename": url,
                "subfolder": "",
                "type": "output"
            })

        return {}
