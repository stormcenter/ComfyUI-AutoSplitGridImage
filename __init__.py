from .GridImageSplitter import GridImageSplitter
from .EvenImageResizer import EvenImageResizer, NODE_CLASS_MAPPINGS as EVEN_RESIZER_NODE_CLASS_MAPPINGS

__all__ = ['GridImageSplitter', 'EvenImageResizer']

NODE_CLASS_MAPPINGS = {
    "GridImageSplitter": GridImageSplitter,
    "EvenImageResizer": EvenImageResizer
}