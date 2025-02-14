# Standard library imports
from importlib.resources import files
from typing import Annotated, Union

# Third-party imports
import torch
from torch import Tensor
from omegaconf import DictConfig, OmegaConf
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


class WavLMSV:
    """
    WavLMSV is a class designed to load a WavLM XVector model from the
    Hugging Face Transformers library for speaker verification or
    diarization tasks. It extracts speaker embeddings from an audio
    waveform using the loaded model. By default, it loads configuration
    parameters from a `config.yaml` file, or you can provide a custom
    `DictConfig`.

    Parameters
    ----------
    config : omegaconf.DictConfig, optional
        Configuration object containing model and runtime parameters.
        If not provided, a default config is loaded from the local files.

    Attributes
    ----------
    device : str
        The device used for inference (e.g., 'cuda' or 'cpu').
    feature_extractor : transformers.Wav2Vec2FeatureExtractor
        The feature extractor used to process raw audio into model inputs.
    model : transformers.WavLMForXVector
        The pretrained WavLM XVector model used for generating embeddings.
    xvector_dim : int
        The dimension of the model's output embeddings.

    Methods
    -------
    extract(waveform, sampling_rate=16000)
        Extract speaker embeddings from the provided waveform.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.create({
    ...     'model': {
    ...         'wavlm': {
    ...             'selected': 'microsoft/wavlm-base-plus-sv',
    ...             'allowed': ['microsoft/wavlm-base-plus-sv']
    ...         }
    ...     },
    ...     'runtime': {
    ...         'device': ['cpu']
    ...     }
    ... })
    >>> embedder = WavLMSV(config=config)
    >>> import torch
    >>> waveform = torch.randn(1, 16000)
    >>> embedding = embedder.extract(waveform)
    >>> print(embedding.shape)
    torch.Size([256])
    """

    def __init__(
            self,
            config: Annotated[Union[DictConfig, None],
            "Configuration for the model and runtime"] = None
    ) -> None:
        """
        Initialize the WavLMSV object.

        Loads the specified WavLM XVector model and sets the appropriate
        device for inference. If no config is provided, the default config
        is loaded from local resources.

        Parameters
        ----------
        config : DictConfig, optional
            Configuration object containing model and runtime parameters.
            If not provided, a default config is loaded from `config.yaml`.

        Raises
        ------
        TypeError
            If `config` is provided but is not a `DictConfig`.
        ValueError
            If the selected model is not in the list of allowed models.
        """
        if config is not None and not isinstance(config, DictConfig):
            raise TypeError("Expected 'DictConfig' for parameter 'config'.")

        if config is None:
            default_config = files("wavlmmsdd.audio.config") / "config.yaml"
            config = OmegaConf.load(str(default_config))

        device_list = config.runtime.device
        device_option = device_list[0]

        if device_option == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA is not available. Falling back to CPU.")
            device_option = "cpu"

        self.device = device_option

        model_name = config.model.wavlm.selected
        allowed_models = config.model.wavlm.allowed
        if model_name not in allowed_models:
            raise ValueError(
                f"{model_name} is not supported in this project. "
                f"Allowed models are: {allowed_models}"
            )

        print(f"[INFO] Loading XVector model: {model_name} on device: {self.device}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.xvector_dim = self.model.config.xvector_output_dim
        print(f"[INFO] XVector dimension: {self.xvector_dim}")

    def extract(
            self,
            waveform: Annotated[Tensor, "Input audio waveform"],
            sampling_rate: Annotated[int, "Sample rate of the waveform"] = 16000
    ) -> Annotated[Tensor, "Speaker embedding vector"]:
        """
        Extract speaker embeddings from the provided waveform.

        The waveform is first converted to mono if needed, and resampled
        to 16 kHz before being passed to the model. The method returns a
        single speaker embedding vector.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform to process. Expected shape is [channels, samples].
        sampling_rate : int, optional
            The sample rate of the waveform. Default is 16 kHz.

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the extracted speaker embedding.

        Raises
        ------
        TypeError
            If `waveform` is not a torch.Tensor, or `sampling_rate`
            is not an int.

        Examples
        --------
        >>> import torch
        >>> embedder = WavLMSV()
        >>> wave = torch.randn(2, 48000)
        >>> emb = embedder.extract(wave, sampling_rate=48000)
        >>> emb.shape
        torch.Size([256])
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Expected 'waveform' to be a torch.Tensor.")
        if not isinstance(sampling_rate, int):
            raise TypeError("Expected 'sampling_rate' to be an int.")

        wave_mono = Convert(waveform=waveform, sample_rate=sampling_rate).to_mono()
        wave_16k, sr_16k = Resample(
            waveform=wave_mono,
            sample_rate=sampling_rate
        ).to_16k()

        inputs = self.feature_extractor(
            wave_16k.squeeze(0).cpu().numpy(),
            sampling_rate=sr_16k,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.embeddings.squeeze(0)

        return embedding.cpu().detach()


if __name__ == "__main__":
    # Local imports
    from wavlmmsdd.audio.preprocess.resample import Resample
    from wavlmmsdd.audio.preprocess.convert import Convert
    from wavlmmsdd.audio.utils.utils import Build
    from wavlmmsdd.audio.diarization.diarize import Diarizer
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
    manifest_path = builder.manifest()

    # Embedding
    test_embedder = WavLMSV()

    # Diarization
    diarizer = Diarizer(embedding=test_embedder, manifest_path=manifest_path)
    diarizer.run()
