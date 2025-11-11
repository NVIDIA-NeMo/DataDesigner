from data_designer.cli.services.provider_service import ProviderService
from data_designer.config.models import ModelProvider


def test_list_all(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    assert stub_provider_service.list_all() == stub_model_providers


def test_get_by_name(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    assert stub_provider_service.get_by_name("test-provider-1") == stub_model_providers[0]
    assert stub_provider_service.get_by_name("test-provider-3") is None


def test_add(
    stub_provider_service: ProviderService,
    stub_model_providers: list[ModelProvider],
    stub_new_model_provider: ModelProvider,
):
    stub_provider_service.add(stub_new_model_provider)
    assert stub_provider_service.list_all() == stub_model_providers + [stub_new_model_provider]


def test_update(stub_provider_service: ProviderService, stub_new_model_provider: ModelProvider):
    stub_provider_service.update("test-provider-1", stub_new_model_provider)
    assert stub_provider_service.get_by_name("test-provider-1") is None
    assert stub_provider_service.get_by_name("test-provider-3") == stub_new_model_provider


def test_delete(stub_provider_service: ProviderService):
    stub_provider_service.delete("test-provider-1")
    assert len(stub_provider_service.list_all()) == 1


def test_set_default(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    stub_provider_service.set_default("test-provider-2")
    assert stub_provider_service.get_default() == "test-provider-2"


def test_get_default(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    assert stub_provider_service.get_default() == "test-provider-1"
