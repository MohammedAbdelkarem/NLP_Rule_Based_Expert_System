from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    LayerParameter,
    OptionSelection,
    TensorInfo,
    Layer,
    WeightShape,
    LayerMetadata,
    TransformerStage,
    CalculationStep
)


class PositionWiseLinearCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="position_wise_linear",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="model_dim",
            value=MATCH.input_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim",
            value=MATCH.output_dim
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            value=MATCH.output_data_type,
            status="resolved"
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_position_wise_linear_layer(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        input_dim,
        output_dim,
        output_data_type
    ):
        input_shape = f"({batch_size}, {sequence_length}, {input_dim})"
        output_shape = f"({batch_size}, {sequence_length}, {output_dim})"
        parameter_shape = f"W: ({input_dim}, {output_dim})\nb: ({output_dim},)"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            weight_name="linear_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="position_wise_linear_output",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type=output_data_type,
            source="position_wise_linear_projection"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source="position_wise_linear_input_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source="position_wise_linear_input_sequence_length"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(output_dim),
            data_type="integer_dimension_value",
            source="position_wise_linear_output_dimension"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            layer_kind="simple_layer",
            layer_role="projection",
            repeat_policy="repeat_allowed",
            consecutive_repeat_policy="allow_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            stage_name="position_wise_linear_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            step_key=f"layer_{layer_index}_position_wise_linear_shape",
            previous_step_key="none",
            step_order=1,
            title="Position-wise Linear output shape",
            formula="Y = XW + b",
            substitution=f"({batch_size}, {sequence_length}, {input_dim}) × ({input_dim}, {output_dim})",
            result=f"{output_shape} as {output_data_type}"
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="position_wise_linear",
            layer_title=MATCH.layer_title
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.previous_output_shape,
            data_type="float_hidden_tensor"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="model_dim",
            value=MATCH.input_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim",
            value=MATCH.output_dim
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            value=MATCH.output_data_type,
            status="resolved"
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_position_wise_linear_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        batch_size,
        sequence_length,
        input_dim,
        output_dim,
        output_data_type
    ):
        input_shape = previous_output_shape
        output_shape = f"({batch_size}, {sequence_length}, {output_dim})"
        parameter_shape = f"W: ({input_dim}, {output_dim})\nb: ({output_dim},)"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            weight_name="linear_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="position_wise_linear_output",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type=output_data_type,
            source="position_wise_linear_projection"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_sequence_length"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(output_dim),
            data_type="integer_dimension_value",
            source="position_wise_linear_output_dimension"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            layer_kind="simple_layer",
            layer_role="projection",
            repeat_policy="repeat_allowed",
            consecutive_repeat_policy="allow_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            stage_name="position_wise_linear_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            step_key=f"layer_{layer_index}_position_wise_linear_shape",
            previous_step_key="none",
            step_order=1,
            title="Position-wise Linear output shape",
            formula="Y = XW + b",
            substitution=f"Layer {previous_layer_index} output {previous_output_shape} × ({input_dim}, {output_dim})",
            result=f"{output_shape} as {output_data_type}"
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
