import os

import pooch

from .birefnet_general import BiRefNetSessionGeneral


class BiRefNetSessionGeneralLite(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-General-Lite session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the BiRefNet-General-Lite model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
            (
                None
                if cls.checksum_disabled(*args, **kwargs)
                else "md5:4fab47adc4ff364be1713e97b7e66334"
            ),
            fname=fname,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-General-Lite session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-general-lite"
