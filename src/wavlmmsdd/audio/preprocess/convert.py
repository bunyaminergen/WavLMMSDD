# Standard library imports
import os
import time
from typing import Annotated, Union

# Third-party imports
import torch
import torchaudio


class Convert:
    """
    Convert is a class for converting an audio waveform to mono
    and saving it to a file.

    This class can be initialized either by providing an audio file path
    (from which the waveform and sample rate will be loaded) or by providing
    both a waveform tensor and its sample rate. The main functionality
    involves converting the audio to a single (mono) channel.

    Parameters
    ----------
    audio_file : str, optional
        Path to an audio file. If provided, this file will be loaded
        for conversion.
    waveform : torch.Tensor, optional
        Audio waveform as a PyTorch tensor. Must be used in conjunction
        with `sample_rate`.
    sample_rate : int, optional
        Sample rate of the provided waveform. Must be used in
        conjunction with `waveform`.

    Attributes
    ----------
    audio_file : str or None
        The path to the audio file (if provided).
    waveform : torch.Tensor
        The loaded or provided audio waveform.
    sample_rate : int
        The loaded or provided waveform's sample rate.

    Methods
    -------
    to_mono() -> torch.Tensor
        Convert the waveform to a single (mono) channel if it is
        not already mono.
    save(output_path: str = None) -> str
        Save the waveform to a specified path or a generated path.

    Examples
    --------
    >>> # Example 1: Using an audio file
    >>> audio_file = "path/to/audio.wav"
    >>> converter = Convert(audio_file=audio_file)
    >>> mono_waveform = converter.to_mono()
    >>> saved_file = converter.save()
    >>> print(saved_file)

    >>> # Example 2: Using a waveform tensor
    >>> wave = torch.randn(2, 48000)  # 2 channels, 48000 samples each
    >>> sr = 48000
    >>> converter = Convert(waveform=wave, sample_rate=sr)
    >>> mono_waveform = converter.to_mono()
    >>> saved_file = converter.save(".temp/mono_example.wav")
    >>> print(saved_file)
    """

    def __init__(
            self,
            audio_file: Annotated[Union[str, None],
            "Path to an audio file or None"] = None,
            waveform: Annotated[Union[torch.Tensor, None],
            "Audio waveform tensor or None"] = None,
            sample_rate: Annotated[Union[int, None],
            "Sample rate or None"] = None
    ) -> None:
        """
        Initialize the Convert object.

        Depending on the provided parameters, either:
        - Load the audio file to get the waveform and sample rate
        - Assign the provided waveform and sample rate directly

        Parameters
        ----------
        audio_file : str, optional
            Path to the audio file to be loaded.
        waveform : torch.Tensor, optional
            Audio waveform tensor. Must provide `sample_rate` if used.
        sample_rate : int, optional
            Sample rate corresponding to the waveform. Must provide
            `waveform` if used.

        Raises
        ------
        TypeError
            If `audio_file` is provided but not a string,
            or if `waveform` is provided but not a tensor,
            or if `sample_rate` is provided but not an integer.
        ValueError
            If neither `audio_file` nor (`waveform` + `sample_rate`)
            is provided.
        """
        if audio_file is not None:
            if not isinstance(audio_file, str):
                raise TypeError("Expected 'str' for parameter 'audio_file'.")
            if not os.path.isfile(audio_file):
                raise FileNotFoundError(
                    f"Audio file not found: {audio_file}"
                )
            wave, sr = torchaudio.load(audio_file)
            self.audio_file = audio_file
            self.waveform = wave
            self.sample_rate = sr

        elif waveform is not None and sample_rate is not None:
            if not isinstance(waveform, torch.Tensor):
                raise TypeError(
                    "Expected 'torch.Tensor' for parameter 'waveform'."
                )
            if not isinstance(sample_rate, int):
                raise TypeError(
                    "Expected 'int' for parameter 'sample_rate'."
                )
            self.audio_file = None
            self.waveform = waveform
            self.sample_rate = sample_rate

        else:
            raise ValueError(
                "Either 'audio_file' or ('waveform' + 'sample_rate') "
                "must be provided."
            )

    def to_mono(self) -> Annotated[torch.Tensor, "Mono waveform"]:
        """
        Convert the waveform to a single (mono) channel.

        If the waveform is already single-channel, it will be
        reshaped if necessary. If it has more than one channel,
        the channels are averaged.

        Returns
        -------
        torch.Tensor
            The mono audio waveform.

        Examples
        --------
        >>> wave = torch.randn(2, 48000)
        >>> sr = 48000
        >>> converter = Convert(waveform=wave, sample_rate=sr)
        >>> mono_waveform = converter.to_mono()
        >>> mono_waveform.shape
        torch.Size([1, 48000])
        """
        if self.waveform.ndim == 1:
            # Reshape the waveform if it's a single 1D channel
            self.waveform = self.waveform.unsqueeze(0)

        if self.waveform.shape[0] > 1:
            # If multiple channels, convert by averaging
            self.waveform = self.waveform.mean(dim=0, keepdim=True)

        return self.waveform

    def save(
            self,
            output_path: Annotated[Union[str, None],
            "Output file path or None"] = None
    ) -> Annotated[str, "Path to the saved audio file"]:
        """
        Save the (possibly mono) waveform to a specified or generated path.

        If `output_path` is not provided, a default path is generated in
        the `.temp/` directory, combining a base name, epoch time, and
        `_mono.wav` suffix.

        Parameters
        ----------
        output_path : str, optional
            Desired output file path. If none is provided, a default
            will be generated in the `.temp/` directory.

        Returns
        -------
        str
            The path where the waveform was saved.

        Raises
        ------
        TypeError
            If `output_path` is provided but is not a string.

        Examples
        --------
        >>> converter = Convert(waveform=torch.randn(1, 16000), sample_rate=16000)
        >>> converter.to_mono()
        >>> saved_file = converter.save(".temp/mono_example.wav")
        >>> print(saved_file)
        .temp/mono_example.wav
        """
        if output_path is not None and not isinstance(output_path, str):
            raise TypeError("Expected 'str' for parameter 'output_path'.")

        if not output_path:
            if self.audio_file:
                base_name = os.path.splitext(
                    os.path.basename(self.audio_file)
                )[0]
            else:
                base_name = "audio"

            epoch_time = int(time.time())
            output_path = f".temp/{base_name}_{epoch_time}_mono.wav"

        dir_name = os.path.dirname(output_path)
        os.makedirs(dir_name, exist_ok=True)

        torchaudio.save(output_path, self.waveform, self.sample_rate)
        print(f"[Convert.save] 16k+mono file saved: {output_path}")
        return output_path


if __name__ == "__main__":
    audio = ".data/example/ae.wav"
    test_converter = Convert(audio_file=audio)
    test_converter.to_mono()
    output = test_converter.save()
    print(f"Saved file: {output}")

    test_waveform = torch.randn(2, 48000)
    rate = 48000
    converter_direct = Convert(waveform=test_waveform, sample_rate=rate)
    converter_direct.to_mono()
    output_path_direct = converter_direct.save(".temp/mono_example.wav")
    print(f"Saved file: {output_path_direct}")
