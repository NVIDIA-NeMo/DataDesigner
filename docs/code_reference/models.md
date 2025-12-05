# Models

The `models` module defines configuration objects for model-based generation. [ModelProvider](#data_designer.config.models.ModelProvider), specifies connection and authentication details for custom providers. [ModelConfig](#data_designer.config.models.ModelConfig) encapsulates model details including the model alias, identifier, and inference parameters. [InferenceParameters](#data_designer.config.models.InferenceParameters) controls model behavior through settings like `temperature`, `top_p`, and `max_tokens`, with support for both fixed values and distribution-based sampling. The module includes [ImageContext](#data_designer.config.models.ImageContext) for providing image inputs to multimodal models.

For more information on how they are used, see below:

- **[Model Providers](../concepts/models/model-providers.md)**
- **[Model Configs](../concepts/models/model-configs.md)**
- **[Image Context](/notebooks/4-providing-images-as-context/)**

::: data_designer.config.models
