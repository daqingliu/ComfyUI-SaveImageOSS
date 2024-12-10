import io
import os
import torch
import numpy as np
import boto3
import json
import tempfile
from PIL import Image, ImageSequence, ImageOps
from PIL.PngImagePlugin import PngInfo


def awss3_load_file(client, bucket, key):
    outfile = io.BytesIO()
    client.download_fileobj(bucket, key, outfile)
    outfile.seek(0)
    return outfile


def awss3_upload_file(client, bucket, local_path, s3_path):
    client.upload_file(local_path, bucket, s3_path)
    return s3_path


def awss3_init_client(region="us-east-1", ak=None, sk=None, endpoint_url=None):
    client = None
    if (ak == None and sk == None) and endpoint_url == None:
        client = boto3.client('s3', region_name=region)
    elif (ak != None and sk != None) and endpoint_url == None:
        client = boto3.client('s3', region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk)
    elif (ak != None and sk != None) and endpoint_url != None:
        client = boto3.client('s3', region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk, endpoint_url=endpoint_url)
    else:
        client = boto3.client('s3')
    return client


# SaveImageToS3
class SaveImageToS3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "images": ("IMAGE",), 
                             "region": ("STRING", {"multiline": False, "default": "us-east-1"}),
                             "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                             "aws_sk": ("STRING", {"multiline": False, "default": ""}),
                             "endpoint_url": ("STRING", {"multiline": False, "default": ""}),
                             "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                             "pathname": ("STRING", {"multiline": False, "default": "pathname for file"}),
                             "save_metadata": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                             },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "save_image_to_s3"
    CATEGORY = "oss/image"
    OUTPUT_NODE = True
    TITLE = "Save Image (S3)"

    def save_image_to_s3(self, images, region, aws_ak, aws_sk, endpoint_url, s3_bucket, pathname, save_metadata, prompt=None, extra_pnginfo=None):
        client = awss3_init_client(region, aws_ak, aws_sk, endpoint_url)
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            if save_metadata:
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file_path = "%s_%i.png"%(pathname, batch_number)
            temp_file = None
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    
                    # Save the image to the temporary file
                    img.save(temp_file_path, pnginfo=metadata, compress_level=0)

                    # Upload the temporary file to S3
                    file_path = awss3_upload_file(client, s3_bucket, temp_file_path, file_path)
                    
                    # Add the result to the results list
                    results.append({
                        "filename": file_path,
                        "subfolder": "",
                        "type": "output"
                    })

            finally:
                # Delete the temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        return {}


# LoadImageFromS3
class LoadImageFromS3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"region": ("STRING", {"multiline": False, "default": "us-east-1"}),
                             "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                             "aws_sk": ("STRING", {"multiline": False, "default": ""}),
                             "endpoint_url": ("STRING", {"multiline": False, "default": ""}),
                             "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                             "pathname": ("STRING", {"multiline": False, "default": "pathname for file"})
                             } 
                }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False, )
    FUNCTION = "load_image_from_s3"
    CATEGORY = "oss/image"

    def load_image_from_s3(self, region, aws_ak, aws_sk, endpoint_url, s3_bucket, pathname):
        client = awss3_init_client(region, aws_ak, aws_sk, endpoint_url)
        img = Image.open(awss3_load_file(client, s3_bucket, pathname))
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

# if __name__ == '__main__':
#     client = awss3_init_client()
#     awss3_save_file(client, "test-bucket", "test.jpg", awss3_load_file(client, "test-bucket", "test"))
