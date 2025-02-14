# Standard library imports
from typing import Annotated

# Local imports
from wavlmmsdd.audio.utils.utils import Build
from wavlmmsdd.audio.feature.embedding import WavLMSV
from wavlmmsdd.audio.preprocess.convert import Convert
from wavlmmsdd.audio.preprocess.resample import Resample
from wavlmmsdd.audio.diarization.diarize import Diarizer


def main() -> Annotated[None, "No return value"]:
    """
    Demonstrate the audio processing workflow from a WAV file
    to a diarization result.

    This function performs the following steps:
    1. Resamples the audio to 16 kHz.
    2. Converts the audio to mono.
    3. Builds a manifest file.
    4. Obtains embeddings.
    5. Runs diarization.

    Returns
    -------
    None

    Examples
    --------
    >>> main()
    No direct output is produced, but the specified audio file is
    processed and the results are saved or printed as logs.
    """

    # Audio Path
    audio_path = ".data/example/ae.wav"

    # Resample to 16 kHz
    resampler = Resample(audio_file=audio_path)
    wave_16k, sr_16k = resampler.to_16k()

    # Convert to Mono
    converter = Convert(waveform=wave_16k, sample_rate=sr_16k)
    converter.to_mono()
    saved_path = converter.save()

    # Build Manifest File
    builder = Build(saved_path)
    manifest_path = builder.manifest()

    # Embedding
    embedder = WavLMSV()

    # Diarization
    diarizer = Diarizer(embedding=embedder, manifest_path=manifest_path)
    diarizer.run()


if __name__ == "__main__":
    main()