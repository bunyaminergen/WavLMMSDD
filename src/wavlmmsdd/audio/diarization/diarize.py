# Standard library imports
from typing import Annotated, Union
from importlib.resources import files

# Third-party imports
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

# Local imports
from wavlmmsdd.audio.feature.embedding import WavLMSV


class Diarizer:
    """
    Diarizer class that performs speaker diarization using NeMo's
    ClusteringDiarizer with custom embeddings from a `WavLMSV` object.

    This class loads a diarization configuration from a YAML file and
    sets up the diarizer to use the provided embedding model. The manifest
    file for diarization is also specified (defaulting to
    `.temp/manifest.json` if not provided).

    Parameters
    ----------
    embedding : WavLMSV, optional
        An instance of the WavLMSV class used to extract embeddings.
        Must not be None.
    manifest_path : str, optional
        Path to the manifest JSON file containing audio metadata.
        Defaults to `.temp/manifest.json` if not provided.

    Attributes
    ----------
    cfg : DictConfig
        Configuration object for diarization loaded from a YAML file.
    clustering_diarizer : nemo.collections.asr.models.ClusteringDiarizer
        The NeMo ClusteringDiarizer initialized with the provided config.
    """

    def __init__(
            self,
            embedding: Annotated[Union[WavLMSV, None],
            "WavLMSV embedding object"] = None,
            manifest_path: Annotated[Union[str, None],
            "Path to the manifest file"] = None
    ) -> None:
        """
        Initialize the Diarizer object with the specified embedding and
        manifest path.

        Parameters
        ----------
        embedding : WavLMSV, optional
            WavLMSV object for generating speaker embeddings.
        manifest_path : str, optional
            Path to the JSON manifest file containing audio metadata.

        Raises
        ------
        TypeError
            If `embedding` is not a `WavLMSV` instance.
        ValueError
            If `embedding` is None.
        TypeError
            If `manifest_path` is provided but is not a string.

        Examples
        --------
        >>> from wavlmmsdd.audio.feature.embedding import WavLMSV
        >>> embed = WavLMSV()
        >>> diarizer = Diarizer(embedding=embed, manifest_path="manifest.json")
        """
        if embedding is None:
            raise ValueError(
                "The 'embedding' parameter cannot be None. "
                "A WavLMSV object is expected."
            )
        if not isinstance(embedding, WavLMSV):
            raise TypeError("Expected 'embedding' to be a 'WavLMSV' instance.")

        if manifest_path is not None and not isinstance(manifest_path, str):
            raise TypeError("Expected 'manifest_path' to be a string if provided.")

        default_config = files("wavlmmsdd.audio.config") / "diar_infer_telephonic.yaml"
        diar_config_path = str(default_config)

        if manifest_path is None:
            manifest_path = ".temp/manifest.json"

        self.cfg = OmegaConf.load(diar_config_path)
        self.cfg.diarizer.manifest_filepath = manifest_path

        self.clustering_diarizer = ClusteringDiarizer(cfg=self.cfg)
        self.clustering_diarizer.speaker_embeddings = embedding

    def run(self) -> None:
        """
        Perform the diarization process using the initialized ClusteringDiarizer.

        Returns
        -------
        None

        Examples
        --------
        >>> from wavlmmsdd.audio.feature.embedding import WavLMSV
        >>> from wavlmmsdd.audio.diarization.diarize import Diarizer
        >>> embed = WavLMSV()
        >>> diarizer = Diarizer(embedding=embed, manifest_path="manifest.json")
        >>> diarizer.run()
        Diarization Completed!
        """
        self.clustering_diarizer.diarize()
        print("Diarization Completed!")


if __name__ == "__main__":
    # Local imports
    from wavlmmsdd.audio.utils.utils import Build
    from wavlmmsdd.audio.feature.embedding import WavLMSV
    from wavlmmsdd.audio.preprocess.convert import Convert
    from wavlmmsdd.audio.preprocess.resample import Resample
    from wavlmmsdd.audio.feature.embedding import WavLMSV

    # Audio Path
    audio_path = ".data/example/ae.wav"

    # Resample to 16 kHz
    resampler = Resample(audio_file=audio_path)
    test_wave_16k, test_sr_16k = resampler.to_16k()

    # Convert to Mono
    converter = Convert(waveform=test_wave_16k, sample_rate=test_sr_16k)
    converter.to_mono()
    saved_path = converter.save()

    # Build Manifest File
    builder = Build(saved_path)
    test_manifest_path = builder.manifest()

    # Embedding
    test_embedder = WavLMSV()

    # Diarization
    test_diarizer = Diarizer(embedding=test_embedder, manifest_path=test_manifest_path)
    test_diarizer.run()
