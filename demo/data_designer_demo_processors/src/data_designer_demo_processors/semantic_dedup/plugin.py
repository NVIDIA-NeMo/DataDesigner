from data_designer.plugins.plugin import Plugin, PluginType

semantic_dedup_plugin = Plugin(
    config_qualified_name="data_designer_demo_processors.semantic_dedup.config.SemanticDedupProcessorConfig",
    impl_qualified_name="data_designer_demo_processors.semantic_dedup.impl.SemanticDedupProcessor",
    plugin_type=PluginType.PROCESSOR,
)
