# Standard library imports
from typing import Annotated, Tuple, Union

# Third-party imports
import torch
import torchaudio


class Resample:
    """
    Resample is a class for converting an audio waveform to 16 kHz.

    This class can be initialized either by providing an audio file path
    (from which the waveform and sample rate will be loaded) or by providing
    both a waveform tensor and its sample rate. The main functionality is to
    resample the audio to 16 kHz if it is not already at that rate.

    Parameters
    ----------
    audio_file : str, optional
        Path to an audio file. If provided, the file is loaded for resampling.
    waveform : torch.Tensor, optional
        Audio waveform as a PyTorch tensor. Must also provide `sample_rate`
        if this parameter is used.
    sample_rate : int, optional
        Sample rate of the provided waveform. Must also provide `waveform`
        if this parameter is used.

    Attributes
    ----------
    waveform : torch.Tensor
        The loaded or provided audio waveform.
    sample_rate : int
        The loaded or provided waveform's sample rate.

    Methods
    -------
    to_16k() -> (torch.Tensor, int)
        Resample the waveform to 16 kHz if needed and return
        the updated waveform and its sample rate.

    Examples
    --------
    >>> # Example 1: Using an audio file
    >>> audio_file = "path/to/audio.wav"
    >>> resampler = Resample(audio_file=audio_file)
    >>> new_waveform, new_sr = resampler.to_16k()
    >>> print(new_sr)
    16000

    >>> # Example 2: Using a waveform tensor
    >>> wave = torch.randn(1, 48000)
    >>> sr = 48000
    >>> resampler = Resample(waveform=wave, sample_rate=sr)
    >>> new_waveform, new_sr = resampler.to_16k()
    >>> print(new_sr)
    16000
    """

    def __init__(
            self,
            audio_file: Annotated[Union[str, None],
            "Path to an audio file or None"] = None,
            waveform: Annotated[Union[torch.Tensor, None],
            "Audio waveform or None"] = None,
            sample_rate: Annotated[Union[int, None],
            "Sample rate or None"] = None
    ) -> None:
        """
        Initialize the Resample object.

        Depending on the provided parameters, either:
        - Load the audio file to get waveform and sample rate
        - Assign the provided waveform and sample rate directly

        Parameters
        ----------
        audio_file : str, optional
            Path to an audio file. If provided, it is used to load
            the waveform and sample rate.
        waveform : torch.Tensor, optional
            Audio waveform tensor. Must provide `sample_rate` too.
        sample_rate : int, optional
            Sample rate corresponding to the waveform. Must provide
            `waveform` too.

        Raises
        ------
        ValueError
            If neither `audio_file` nor (`waveform` + `sample_rate`)
            are provided.
        TypeError
            If any provided arguments are not of the expected type.
        """
        if audio_file is not None:
            if not isinstance(audio_file, str):
                raise TypeError("Expected 'str' for parameter 'audio_file'.")
            wave, sr = torchaudio.load(audio_file)
            self.waveform = wave
            self.sample_rate = sr
        elif waveform is not None and sample_rate is not None:
            if not isinstance(waveform, torch.Tensor):
                raise TypeError("Expected 'torch.Tensor' for parameter 'waveform'.")
            if not isinstance(sample_rate, int):
                raise TypeError("Expected 'int' for parameter 'sample_rate'.")
            self.waveform = waveform
            self.sample_rate = sample_rate
        else:
            raise ValueError(
                "Either 'audio_file' or ('waveform' + 'sample_rate') "
                "must be provided."
            )

    def to_16k(
            self
    ) -> Annotated[Tuple[torch.Tensor, int],
    "Resampled waveform and its new sample rate"]:
        """
        Resample the waveform to 16 kHz if needed.

        If the waveform's sample rate is already 16 kHz,
        no resampling is performed.

        Returns
        -------
        (torch.Tensor, int)
            Tuple containing the resampled waveform and its sample rate.

        Examples
        --------
        >>> wave = torch.randn(1, 48000)
        >>> sr = 48000
        >>> test_resampler = Resample(waveform=wave, sample_rate=sr)
        >>> new_waveform, new_sr = test_resampler.to_16k()
        >>> print(new_waveform.shape, new_sr)
        torch.Size([1, 16000]) 16000
        """
        if self.sample_rate == 16000:
            return self.waveform, self.sample_rate

        resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate)
        self.waveform = resampler(self.waveform)
        self.sample_rate = 16000
        return self.waveform, self.sample_rate


if __name__ == "__main__":
    audio = ".data/example/ae.wav"

    resample_instance = Resample(audio_file=audio)
    test_new_waveform, test_new_sr = resample_instance.to_16k()
    print(f"Resampled waveform shape: {test_new_waveform.shape}, sample rate: {test_new_sr}")

    test_wave = torch.randn(1, 32000)
    test_sr: int = 48000
    test_resample_instance = Resample(waveform=test_wave, sample_rate=test_sr)
    new_waveform_test, new_sr_test = resample_instance.to_16k()
    print(f"Resampled waveform shape: {new_waveform_test.shape}, sample rate: {new_sr_test}")
