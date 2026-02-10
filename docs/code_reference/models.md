# Models

The `models` module defines configuration objects for model-based generation. [ModelProvider](#data_designer.config.models.ModelProvider) specifies connection and authentication details for custom providers. [ModelConfig](#data_designer.config.models.ModelConfig) encapsulates model details including the model alias, identifier, and inference parameters. [Inference Parameters](../concepts/models/inference-parameters.md) controls model behavior through settings like `temperature`, `top_p`, and `max_tokens`, with support for both fixed values and distribution-based sampling. The module includes [ImageContext](#data_designer.config.models.ImageContext) for providing image inputs to multimodal models, and [ImageInferenceParams](#data_designer.config.models.ImageInferenceParams) for configuring image generation models.

For more information on how they are used, see below:

- **[Model Providers](../concepts/models/model-providers.md)**
- **[Model Configs](../concepts/models/model-configs.md)**
- **[Image Context](../notebooks/4-providing-images-as-context.ipynb)**
- **[Generating Images](../notebooks/5-generating-images.ipynb)**

::: data_designer.config.models
