""" Base Session Class """
import os
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage
from folder_paths import models_dir

class BaseSession:
    """This is a base class for managing a session with a machine learning model."""

    def __init__(
        self,
        model_name: str,
        sess_opts: ort.SessionOptions,
        providers=None,
        *args,
        **kwargs
    ):
        """Initialize an instance of the BaseSession class."""
        self.model_name = model_name

        self.providers = []

        if providers:
            if type(providers) == list:
                self.providers = providers
            else:
                self.providers.append(providers)
        
        self.inner_session = ort.InferenceSession(
            str(self.__class__.download_models(*args, **kwargs)),
            providers=self.providers,
            sess_options=sess_opts,
        )

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
        *args,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """ Normalizes and resizes image to be used in prediction """
        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """Predict using the loaded model"""
        raise NotImplementedError

    @classmethod
    def checksum_disabled(cls, *args, **kwargs):
        """Checks whether checksums are disabled"""
        return os.getenv("MODEL_CHECKSUM_DISABLED", None) is not None

    @classmethod
    def u2net_home(cls, *args, **kwargs):
        """Returns U2Net home directory"""
        return os.path.join(models_dir, 'birefnet')
        # return os.path.expanduser(
        #     os.getenv(
        #         "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
        #     )
        # )

    @classmethod
    def download_models(cls, *args, **kwargs):
        """Download models"""
        raise NotImplementedError

    @classmethod
    def name(cls, *args, **kwargs):
        """Model Name"""
        raise NotImplementedError
