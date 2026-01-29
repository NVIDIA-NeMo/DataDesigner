# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph builder for constructing execution graphs from DataDesigner configs.

This module provides the GraphBuilder class that constructs ExecutionGraph
instances from DataDesignerConfig objects. It infers execution traits from
generator properties (not class names) to support plugin generators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.config.base import ConfigBase
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
)
from data_designer.engine.column_generators.registry import ColumnGeneratorRegistry
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.utils.config_compiler import compile_dataset_builder_column_configs
from data_designer.engine.execution_graph.column_descriptor import ColumnDescriptor
from data_designer.engine.execution_graph.graph import ExecutionGraph
from data_designer.engine.execution_graph.traits import ExecutionTraits

if TYPE_CHECKING:
    from data_designer.config.data_designer_config import DataDesignerConfig


class GraphBuilder:
    """Factory for constructing ExecutionGraph instances from DataDesigner configs.

    The GraphBuilder infers execution traits from generator properties (not class
    names) to support plugin generators. It handles multi-column configs by marking
    additional columns as side effects.

    Example:
        >>> builder = GraphBuilder(column_generator_registry)
        >>> graph = builder.build(config, num_records=1_000_000)
        >>> for node in graph.iter_start_nodes():
        ...     print(node)
    """

    def __init__(self, registry: ColumnGeneratorRegistry) -> None:
        """Initialize the graph builder.

        Args:
            registry: The column generator registry to use for looking up generators.
        """
        self._registry = registry

    def build(
        self,
        config: DataDesignerConfig,
        num_records: int,
        *,
        strict: bool = True,
    ) -> ExecutionGraph:
        """Build an execution graph from a DataDesigner config.

        Args:
            config: The DataDesigner configuration.
            num_records: The number of records to generate.
            strict: If True, validate all dependencies exist during construction.

        Returns:
            An ExecutionGraph ready for async execution.

        Raises:
            ValueError: If no columns have the START trait (can generate from scratch).
        """
        descriptors = self._build_column_descriptors(config)

        # Validate at least one start column exists
        has_start = any(desc.is_start_column for desc in descriptors)
        if not has_start:
            raise ValueError(
                "At least one column must be able to generate from scratch (have can_generate_from_scratch=True)"
            )

        return ExecutionGraph(num_records, descriptors, strict=strict)

    def _build_column_descriptors(self, config: DataDesignerConfig) -> list[ColumnDescriptor]:
        """Build column descriptors from config.

        This method compiles the user-facing column configs (e.g., SamplerColumnConfig)
        into internal multi-column configs (e.g., SamplerMultiColumnConfig) that the
        registry expects, then builds descriptors from those compiled configs.

        Args:
            config: The DataDesigner configuration.

        Returns:
            List of ColumnDescriptor objects in topological order.
        """
        # Compile user-facing configs into internal multi-column configs
        compiled_configs = compile_dataset_builder_column_configs(config)

        descriptors: list[ColumnDescriptor] = []
        for col_config in compiled_configs:
            descriptor = self._build_column_descriptor(col_config)
            descriptors.append(descriptor)

        return descriptors

    def _build_column_descriptor(self, col_config: ConfigBase) -> ColumnDescriptor:
        """Build a single column descriptor from a config.

        Args:
            col_config: The column configuration (SingleColumnConfig or MultiColumnConfig).

        Returns:
            A ColumnDescriptor for the column.
        """
        gen_cls = self._registry.get_for_config_type(type(col_config))
        traits = self._infer_traits(gen_cls)

        if isinstance(col_config, MultiColumnConfig):
            # Multi-column configs use the first column as primary name
            # Additional columns are marked as side effects
            primary_name = col_config.columns[0].name
            additional_columns = [c.name for c in col_config.columns[1:]]

            return ColumnDescriptor(
                name=primary_name,
                config=col_config,
                generator_cls=gen_cls,
                traits=traits,
                dependencies=[],  # Multi-column configs typically have no dependencies
                side_effects=additional_columns,
            )

        # Single column config
        return ColumnDescriptor(
            name=col_config.name,
            config=col_config,
            generator_cls=gen_cls,
            traits=traits,
            dependencies=col_config.required_columns,
            side_effects=col_config.side_effect_columns,
        )

    def _infer_traits(self, gen_cls: type[ColumnGenerator]) -> ExecutionTraits:
        """Infer execution traits from generator class properties.

        This method uses generator properties (not class names) to determine
        execution traits, making it compatible with plugin generators.

        Args:
            gen_cls: The generator class to analyze.

        Returns:
            ExecutionTraits flags for the generator.
        """
        traits = ExecutionTraits.NONE

        # Check can_generate_from_scratch - use getattr for simple property access
        # This works for both class attributes and properties with simple True/False returns
        can_generate = getattr(gen_cls, "can_generate_from_scratch", False)
        # Handle property objects vs actual values
        if isinstance(can_generate, property):
            # For properties, check if the class overrides with a known pattern
            can_generate = self._evaluate_property_default(gen_cls, "can_generate_from_scratch", False)
        if can_generate:
            traits |= ExecutionTraits.START

        # Check generation strategy
        strategy = gen_cls.get_generation_strategy()
        if strategy == GenerationStrategy.CELL_BY_CELL:
            traits |= ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE
        else:  # FULL_COLUMN
            # Check is_row_streamable property
            is_streamable = getattr(gen_cls, "is_row_streamable", False)
            if isinstance(is_streamable, property):
                is_streamable = self._evaluate_property_default(gen_cls, "is_row_streamable", False)
            if is_streamable:
                traits |= ExecutionTraits.ROW_STREAMABLE
            else:
                traits |= ExecutionTraits.BARRIER

        return traits

    def _evaluate_property_default(
        self,
        cls: type[ColumnGenerator],
        property_name: str,
        default: bool,
    ) -> bool:
        """Evaluate the default value of a property.

        For simple properties that return True or False, this inspects the
        bytecode to determine the return value without instantiating.

        Args:
            cls: The class to check.
            property_name: The name of the property.
            default: The default value if the property cannot be determined.

        Returns:
            The property's default value.
        """
        try:
            import dis

            prop = getattr(cls, property_name, None)
            if not isinstance(prop, property) or prop.fget is None:
                return default

            fget = prop.fget
            code = fget.__code__
            instructions = list(dis.get_instructions(code))

            for i, instr in enumerate(instructions):
                # Python 3.13+ uses RETURN_CONST for simple constant returns
                if instr.opname == "RETURN_CONST" and isinstance(instr.argval, bool):
                    return instr.argval

                # Pre-3.13: LOAD_CONST True/False followed by RETURN_VALUE
                if instr.opname == "RETURN_VALUE" and i > 0:
                    prev = instructions[i - 1]
                    if prev.opname == "LOAD_CONST" and isinstance(prev.argval, bool):
                        return prev.argval

            # If not a simple constant return, check based on property name
            if property_name == "is_row_streamable":
                # Default implementation compares generation strategy
                strategy = cls.get_generation_strategy()
                return strategy == GenerationStrategy.CELL_BY_CELL

            if property_name == "can_generate_from_scratch":
                # Default is False in base ColumnGenerator
                return default

        except Exception:
            pass

        return default
