from .DocumentConverter import DocumentConverter
from typing import Dict, Type
from .converters.BaseConverter import BaseConverter


class LoaderFactory:
    """
    Registry-based factory to get a converter for a given file extension.
    Register converters with register(extension, ConverterClass).
    """
    _registry: Dict[str, Type[DocumentConverter]] = {}

    @classmethod
    def register(cls, ext: str, converter_cls: Type[DocumentConverter]):
        cls._registry[ext.lower()] = converter_cls

    @classmethod
    def get_converter(cls, ext: str) -> DocumentConverter:
        ext = (ext or "").lower()
        if ext in cls._registry:
            return cls._registry[ext]()
        # try common mapping if user passed with/without leading dot
        if ext.startswith(".") and ext[1:] in cls._registry:
            return cls._registry[ext[1:]]()
        # fallback to base
        return BaseConverter()
