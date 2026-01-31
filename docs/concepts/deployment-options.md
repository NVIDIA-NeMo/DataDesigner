# Deployment Options: Library vs. Microservice

Data Designer is available as both an **open-source library** and a **NeMo Microservice**. This guide helps you choose the right deployment option for your use case.

## Quick Comparison

| Aspect | Open-Source Library | NeMo Microservice |
|--------|---------------------|-------------------|
| **What it is** | Python package you import and run | REST API service exposing `preview` and `create` methods |
| **Best for** | Developers with LLM access who want flexibility and customization | Teams using NeMo Microservices platform |
| **LLM Access** | You provide (any OpenAI-compatible API) | Integrated with NeMo Inference Microservices |
| **Installation** | `pip install data-designer` | Deploy via NeMo Microservices platform |
| **Scaling** | You manage inference capacity | Managed alongside other NeMo services |

!!! success "Same Configuration API"
    Both the library and microservice use the **same `DataDesignerConfigBuilder` API**. Start with the library, and your configurations migrate seamlessly if you later adopt the NeMo platform.

## When to Use the Open-Source Library

The library is the right choice for most users. Choose it if you:

### You Have Access to LLMs

You have API keys or endpoints for LLM inference:

- **Cloud APIs**: NVIDIA API Catalog (build.nvidia.com), OpenAI, Azure OpenAI, Anthropic
- **Self-hosted**: vLLM, TGI, TensorRT-LLM, or any OpenAI-compatible server
- **Enterprise gateways**: Centralized LLM gateway with RBAC, rate limiting, or other enterprise features

```python
from data_designer import DataDesigner
from data_designer.config import ModelConfig

# Use any OpenAI-compatible endpoint
model = ModelConfig(
    alias="my-model",
    model="meta/llama-3.1-8b-instruct",
    provider="nvidia",  # or "openai", "azure", custom base_url
)

dd = DataDesigner()
# Your code controls the full workflow
```

### You Need Maximum Flexibility

- **Custom plugins**: Extend Data Designer with custom column generators, validators, or processors
- **Local development**: Rapid iteration with immediate feedback
- **Integration**: Embed Data Designer into existing Python pipelines or notebooks
- **Experimentation**: Research workflows with custom models or configurations

### You Already Have Enterprise LLM Infrastructure

!!! tip "Library + Enterprise LLM Gateway"
    Many enterprises already have centralized LLM access through API gateways with:

    - Role-based access control (RBAC)
    - Rate limiting and quotas
    - Audit logging
    - Cost allocation

    In this case, **use the library** and point it at your enterprise gateway. You get enterprise-grade LLM access while retaining full control over your Data Designer workflows.

```python
# Point Data Designer at your enterprise LLM gateway
model = ModelConfig(
    alias="enterprise-llm",
    model="gpt-4",
    provider="openai",
    base_url="https://llm-gateway.yourcompany.com/v1",  # Your gateway
    api_key_env_var="ENTERPRISE_LLM_KEY",
)
```

---

## When to Use the Microservice

The NeMo Microservice exposes Data Designer's `preview` and `create` methods as REST API endpoints. Choose it if you:

### You're Using the NeMo Microservices Platform

The primary value of the microservice is **integration with other NeMo Microservices**:

- **NeMo Inference Microservices (NIMs)**: Seamless integration with NVIDIA's optimized inference endpoints
- **NeMo Customizer**: Generate synthetic data for model fine-tuning workflows
- **NeMo Evaluator**: Create evaluation datasets alongside model assessment
- **Unified deployment**: Single platform for your entire AI pipeline


### You Want to Expose SDG as a Team Service

If you need to provide synthetic data generation as a shared service:

- **Multi-tenant access**: Multiple teams submit generation jobs via API
- **Job management**: Queue, monitor, and manage generation jobs centrally
- **Resource sharing**: Shared infrastructure for SDG workloads

---

## Decision Flowchart

```
                    ┌─────────────────────────┐
                    │ Are you using the NeMo  │
                    │ Microservices platform? │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                   YES                      NO
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────────────┐
        │ Use Microservice  │   │ Do you need to expose SDG │
        │                   │   │ as a shared REST service? │
        │ Integrates with   │   └─────────────┬─────────────┘
        │ NIMs, Customizer, │                 │
        │ Evaluator         │     ┌───────────┴───────────┐
        └───────────────────┘     ▼                       ▼
                                 YES                      NO
                                  │                       │
                                  ▼                       ▼
                      ┌─────────────────────┐   ┌─────────────────┐
                      │ Consider if the     │   │ Use the Library │
                      │ overhead is worth   │   │                 │
                      │ it vs. library +    │   │ Most flexible   │
                      │ enterprise gateway  │   │ option for      │
                      └─────────────────────┘   │ direct use      │
                                                └─────────────────┘
```

---

## Learn More

- **Library**: Continue with this documentation
- **Microservice**: See the [NeMo Data Designer Microservice documentation](https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/index.html){target="_blank"}
