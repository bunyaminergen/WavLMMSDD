# Standard library imports
from typing import Annotated, List
from dataclasses import dataclass, field


@dataclass
class WavLMModels:
    """
    WavLMModels is a data class that stores configuration parameters
    for WavLM model usage.

    This class provides a list of allowed model names and a selected
    model name for audio processing tasks.

    Parameters
    ----------
    allowed : List[str], optional
        A list of allowed WavLM model names.
    selected : str, optional
        The currently selected WavLM model name.

    Attributes
    ----------
    allowed : List[str]
        A list of allowed WavLM model names.
    selected : str
        The currently selected WavLM model name.

    Examples
    --------
    >>> models = WavLMModels(allowed=["microsoft/wavlm-base", "microsoft/wavlm-base-plus-sv"])
    >>> models.allowed
    ['microsoft/wavlm-base', 'microsoft/wavlm-base-plus-sv']
    >>> models.selected
    'microsoft/wavlm-base-plus-sv'
    """

    allowed: Annotated[List[str], "List of allowed WavLM model names"] = field(default_factory=list)
    selected: Annotated[str, "Currently selected WavLM model name"] = "microsoft/wavlm-base-plus-sv"

    def __post_init__(self):
        """
        Validate the model names after initialization.

        Raises
        ------
        TypeError
            If 'allowed' is not a list of strings or 'selected' is not a string.
        """
        if not isinstance(self.allowed, list):
            raise TypeError("Expected 'allowed' to be a list of strings.")
        for model_name in self.allowed:
            if not isinstance(model_name, str):
                raise TypeError("All items in 'allowed' must be strings.")

        if not isinstance(self.selected, str):
            raise TypeError("Expected 'selected' to be a string.")


@dataclass
class ModelConfig:
    """
    ModelConfig is a data class that wraps the WavLMModels configuration,
    consolidating model-related settings in one place.

    Parameters
    ----------
    wavlm : WavLMModels, optional
        The WavLMModels configuration data class.

    Attributes
    ----------
    wavlm : WavLMModels
        An instance of WavLMModels containing allowed and selected model names.

    Examples
    --------
    >>> from dataclasses import asdict
    >>> models = WavLMModels(allowed=["microsoft/wavlm-base"], selected="microsoft/wavlm-base")
    >>> config = ModelConfig(wavlm=models)
    >>> asdict(config)
    {'wavlm': {'allowed': ['microsoft/wavlm-base'], 'selected': 'microsoft/wavlm-base'}}
    """

    wavlm: Annotated[WavLMModels, "Configuration object for WavLM models"] = WavLMModels()

    def __post_init__(self):
        """
        Validate that the 'wavlm' field is an instance of WavLMModels.

        Raises
        ------
        TypeError
            If 'wavlm' is not an instance of WavLMModels.
        """
        if not isinstance(self.wavlm, WavLMModels):
            raise TypeError("Expected 'wavlm' to be an instance of WavLMModels.")


@dataclass
class RuntimeConfig:
    """
    RuntimeConfig is a data class that stores runtime configuration details
    such as device selection and CUDA allocation settings.

    Parameters
    ----------
    device : List[str], optional
        A prioritized list of devices (e.g., ['cuda', 'cpu']) for running
        audio processing or machine learning tasks.
    cuda_alloc_conf : str, optional
        A string specifying CUDA memory allocation configurations.

    Attributes
    ----------
    device : List[str]
        A list of devices in order of priority.
    cuda_alloc_conf : str
        CUDA memory allocation configuration settings.

    Examples
    --------
    >>> runtime = RuntimeConfig()
    >>> runtime.device
    ['cuda', 'cpu']
    >>> runtime.cuda_alloc_conf
    'expandable_segments:True'
    """

    device: Annotated[List[str], "List of devices in priority order"] = field(
        default_factory=lambda: ["cuda", "cpu"]
    )
    cuda_alloc_conf: Annotated[str, "CUDA memory allocation configuration"] = "expandable_segments:True"

    def __post_init__(self):
        """
        Validate the runtime fields after initialization.

        Raises
        ------
        TypeError
            If 'device' is not a list of strings or 'cuda_alloc_conf' is not a string.
        """
        if not isinstance(self.device, list):
            raise TypeError("Expected 'device' to be a list of strings.")
        for dev in self.device:
            if not isinstance(dev, str):
                raise TypeError("All items in 'device' must be strings.")

        if not isinstance(self.cuda_alloc_conf, str):
            raise TypeError("Expected 'cuda_alloc_conf' to be a string.")


@dataclass
class RootConfig:
    """
    RootConfig is a data class that integrates runtime and model configurations
    into a single configuration structure.

    This class serves as a top-level container for both runtime and model
    settings, making it convenient to manage all configuration options
    in one place.

    Parameters
    ----------
    runtime : RuntimeConfig, optional
        Configuration data class for runtime-specific settings.
    model : ModelConfig, optional
        Configuration data class for model-specific settings.

    Attributes
    ----------
    runtime : RuntimeConfig
        Contains runtime-related configuration (device, CUDA alloc).
    model : ModelConfig
        Contains model-related configuration (WavLM model settings).

    Examples
    --------
    >>> config = RootConfig()
    >>> config.runtime.device
    ['cuda', 'cpu']
    >>> config.model.wavlm.selected
    'microsoft/wavlm-base-plus-sv'
    """

    runtime: Annotated[RuntimeConfig, "Runtime configuration settings"] = RuntimeConfig()
    model: Annotated[ModelConfig, "Model configuration settings"] = ModelConfig()

    def __post_init__(self):
        """
        Validate the root configuration fields after initialization.

        Raises
        ------
        TypeError
            If 'runtime' is not a RuntimeConfig instance or 'model' is not
            a ModelConfig instance.
        """
        if not isinstance(self.runtime, RuntimeConfig):
            raise TypeError("Expected 'runtime' to be a RuntimeConfig instance.")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("Expected 'model' to be a ModelConfig instance.")
