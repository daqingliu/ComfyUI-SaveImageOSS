from .nodes.node_url import SaveImageURL
from .nodes.node_s3 import SaveImageToS3, LoadImageFromS3

NODE_CLASS_MAPPINGS = {
    "SaveImageURL": SaveImageURL,
    "SaveImageS3": SaveImageToS3,
    "LoadImageS3": LoadImageFromS3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageURL": "Save Image to URL",
    "LoadImageS3": "Load Image from S3",
    "SaveImageS3": "Save Image to S3",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]