# Models

[ModelProvider](#data_designer.config.models.ModelProvider) stores connection and authentication details for model providers. [ModelConfig](#data_designer.config.models.ModelConfig) stores a model alias, model identifier, provider settings, and inference parameters. [Inference Parameters](../../concepts/models/inference-parameters.md) control model behavior. Chat-completion parameters include `temperature`, `top_p`, and `max_tokens`; `temperature` and `top_p` can be fixed values or configured distributions. [ImageContext](#data_designer.config.models.ImageContext), [AudioContext](#data_designer.config.models.AudioContext), and [VideoContext](#data_designer.config.models.VideoContext) provide image, audio, and video inputs to multimodal models, and [ImageInferenceParams](#data_designer.config.models.ImageInferenceParams) configures image generation models.

Related guides:

- **[Model Providers](../../concepts/models/model-providers.md)**
- **[Model Configs](../../concepts/models/model-configs.md)**
- **[Image Context](../../notebooks/4-providing-images-as-context.ipynb)**
- **[Generating Images](../../notebooks/5-generating-images.ipynb)**

::: data_designer.config.models
