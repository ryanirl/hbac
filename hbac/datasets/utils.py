import numpy as np
import h5py 

from typing import Tuple
from typing import Dict
from typing import Any


class LazyH5Database:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.db = h5py.File(filepath, "r")

        # We maintain a cache of values as to only load the data into memory once.
        self.cache: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self.cache:
            self.cache[key] = self.pre_cache_hook(self.db[key][()])

        return self.post_cache_hook(self.cache[key])

    def pre_cache_hook(self, value: Any) -> Any:
        return value

    def post_cache_hook(self, value: Any) -> Any:
        return value


class EegDatabase(LazyH5Database):
    """The EEG Database for the `HMS` dataset needs a post_cache_hook to unpack
    the `ekg`, `mid`, and `chain` from a single array.
    """
    def post_cache_hook(self, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ekg = value[0].reshape(1, -1)
        mid = value[1:3].reshape(2, -1)
        eeg = value[3:].reshape(16, -1).reshape(4, 4, -1)

        return eeg, mid, ekg


