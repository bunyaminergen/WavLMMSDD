# Standard library imports
import os
import json
from typing import Annotated

# Third-party imports
import torchaudio


class Build:
    """
    Build class to generate a manifest file for the given audio file.

    This class loads the audio file, calculates its duration, and
    creates a JSON manifest containing metadata for downstream tasks.

    Parameters
    ----------
    audio_path : str
        The path to the audio file.

    Attributes
    ----------
    audio_path : str
        The path to the audio file.
    duration : float
        The audio file's duration in seconds.

    Methods
    -------
    manifest()
        Generate and return the path to the JSON manifest file.

    Examples
    --------
    >>> b = Build("audio.wav")
    >>> path = b.manifest()
    >>> print(path)
    path/to/manifest.json
    """

    def __init__(
            self,
            audio_path: Annotated[str, "Path to the audio file"]
    ) -> None:
        """
        Initialize the Build object with the provided audio file path.

        The constructor verifies that the file exists and loads it
        to determine its duration.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.

        Raises
        ------
        TypeError
            If `audio_path` is not a string.
        FileNotFoundError
            If the file at `audio_path` does not exist.
        """
        if not isinstance(audio_path, str):
            raise TypeError("Expected 'str' for parameter 'audio_path'")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        self.audio_path = audio_path
        waveform, sr = torchaudio.load(self.audio_path)
        num_samples = waveform.shape[1]
        self.duration = num_samples / sr

    def manifest(self) -> Annotated[str, "Path to the JSON manifest file"]:
        """
        Generate a JSON manifest file and return its path.

        This method creates a JSON manifest with fields:
        'audio_filepath', 'offset', 'duration', and 'text'.
        The resulting file is saved in the same directory as
        the audio file.

        Returns
        -------
        str
            The absolute path to the generated JSON manifest file.

        Examples
        --------
        >>> b = Build("audio.wav")
        >>> b.manifest()
        'path/to/manifest.json'
        """
        manifest_data = {
            "audio_filepath": self.audio_path,
            "offset": 0.0,
            "duration": self.duration,
            "text": ""
        }

        audio_dir = os.path.dirname(self.audio_path)
        manifest_path = os.path.join(audio_dir, "manifest.json")

        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_data, ensure_ascii=False))

        return manifest_path


if __name__ == "__main__":
    audio = ".data/example/ae.wav"
    builder = Build(audio)
    result = builder.manifest()
    print(f"Manifest File: {result}")
