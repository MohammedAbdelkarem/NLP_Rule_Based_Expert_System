from experta import Rule, MATCH, NOT

from src.facts import ComponentGroup, Layer, TensorInfo


class DeltaFeatureAdapterRules:
    @Rule(
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="delta_features",
            group_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.separate_output_shape,
            status="computed"
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="delta_features",
            tensor_name="delta_feature_concatenation_adapter",
            role="adapter_output",
            shape=MATCH.adapter_shape,
            data_type=MATCH.adapter_data_type
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=59
    )
    def declare_delta_primary_output_and_available_adapter(
        self,
        layer_index,
        layer_title,
        input_shape,
        separate_output_shape,
        adapter_shape,
        adapter_data_type
    ):
        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="delta_features",
            tensor_name="delta_feature_stream_group",
            role="output",
            dims_count=4,
            shape=separate_output_shape,
            data_type="separate_delta_feature_stream_group",
            source="delta_feature_component_group_primary_output"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="delta_features",
            tensor_name="delta_adapter_suggestion_target",
            role="adapter_suggestion",
            dims_count=4,
            shape=adapter_shape,
            data_type=adapter_data_type,
            source="use_when_next_layer_requires_single_concatenated_feature_tensor"
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="delta_features",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=separate_output_shape
        ))
