# Package: src.ingestion.datasets
from .base import BaseDatasetAdapter
from .zurich_mav import ZurichMAVAdapter

__all__ = [
    "BaseDatasetAdapter",
    "ZurichMAVAdapter"
]
