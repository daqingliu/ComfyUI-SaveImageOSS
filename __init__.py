from .nodes import SaveImageOSS, LoadImageOSS

NODE_CLASS_MAPPINGS = {
    "SaveImageOSS": SaveImageOSS,
    "LoadImageOSS": LoadImageOSS,
}

__all__ = ["NODE_CLASS_MAPPINGS"]