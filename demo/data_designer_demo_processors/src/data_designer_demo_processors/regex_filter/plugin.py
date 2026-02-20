from data_designer.plugins.plugin import Plugin, PluginType

regex_filter_plugin = Plugin(
    config_qualified_name="data_designer_demo_processors.regex_filter.config.RegexFilterProcessorConfig",
    impl_qualified_name="data_designer_demo_processors.regex_filter.impl.RegexFilterProcessor",
    plugin_type=PluginType.PROCESSOR,
)
