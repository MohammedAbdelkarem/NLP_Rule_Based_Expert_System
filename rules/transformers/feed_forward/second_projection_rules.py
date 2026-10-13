from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    OptionSelection,
    LayerParameter,
    IntermediateTensor,
    WeightShape,
    TensorInfo,
    ComponentGroup,
    LayerMetadata,
    TransformerStage,
    CalculationStep,
    Layer
)


class FeedForwardSecondProjectionRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            layer_title=MATCH.layer_title
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            value=MATCH.activation_name,
            status="resolved"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="feed_forward_dim",
            value=MATCH.feed_forward_dim
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            tensor_name="activation_output",
            shape=MATCH.activation_output_shape
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=62
    )
    def calculate_feed_forward_second_projection_shape(
        self,
        layer_index,
        layer_title,
        activation_name,
        batch_size,
        sequence_length,
        model_dim,
        feed_forward_dim,
        activation_output_shape
    ):
        input_shape = f"({batch_size}, {sequence_length}, {model_dim})"
        output_shape = input_shape
        second_projection_shape = output_shape
        parameter_shape = (
            f"W1: ({model_dim}, {feed_forward_dim}), "
            f"b1: ({feed_forward_dim},), "
            f"W2: ({feed_forward_dim}, {model_dim}), "
            f"b2: ({model_dim},)"
        )
        internal_layers = (
            f"First Position-wise Linear Projection + {activation_name.upper()} Activation + "
            "Second Position-wise Linear Projection"
        )
        combine_operation = (
            "The first projection expands model_dim to d_ff, the selected activation preserves d_ff, "
            "and the second projection maps d_ff back to model_dim."
        )

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="second_projection_output",
            shape=second_projection_shape,
            data_type="float_hidden_tensor",
            source="position_wise_linear_projection_W2_b2"
        ))

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="feed_forward",
            weight_name="feed_forward_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="output_feed_forward_tensor",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type="float_hidden_tensor",
            source="second_position_wise_projection"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source="feed_forward_input_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source="feed_forward_input_sequence_length"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(model_dim),
            data_type="integer_dimension_value",
            source="feed_forward_second_projection_dimension"
        ))

        self.declare(ComponentGroup(
            group_index=layer_index,
            group_name="feed_forward",
            group_title="Feed-Forward Network Layer Group",
            internal_layers=internal_layers,
            combine_operation=combine_operation,
            input_shape=input_shape,
            output_shape=output_shape,
            status="computed"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="feed_forward",
            layer_kind="grouped_layer",
            layer_role="feature_transformation",
            repeat_policy="repeat_allowed",
            consecutive_repeat_policy="allow_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="feed_forward",
            stage_name="feed_forward_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="feed_forward",
            step_key=f"layer_{layer_index}_ffn_second_projection_shape",
            previous_step_key=f"layer_{layer_index}_ffn_activation_shape",
            step_order=3,
            title="Feed-Forward second projection shape",
            formula="(batch_size, sequence_length, d_ff) × W2(d_ff, model_dim)",
            substitution=f"{activation_output_shape} × ({feed_forward_dim}, {model_dim})",
            result=second_projection_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="feed_forward",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
