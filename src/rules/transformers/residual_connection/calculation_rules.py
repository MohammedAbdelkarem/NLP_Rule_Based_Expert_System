from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    WeightShape,
    Layer,
    LayerMetadata,
    CalculationStep,
    TransformerStage
)


class ResidualConnectionCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="residual_connection",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=MATCH.source_layer_index
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.previous_shape,
            data_type="float_hidden_tensor"
        ),
        TensorInfo(
            layer_index=MATCH.source_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.previous_shape,
            data_type="float_hidden_tensor"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.batch_size
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.sequence_length
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.model_dim
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_residual_connection_output(
        self,
        layer_index,
        previous_layer_index,
        source_layer_index,
        layer_title,
        previous_shape,
        batch_size,
        sequence_length,
        model_dim
    ):
        input_shape = f"previous_output: {previous_shape}, residual_source_layer_{source_layer_index}: {previous_shape}"
        output_shape = previous_shape
        parameter_shape = "no trainable parameters"

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="residual_connection",
            tensor_name="previous_layer_output_tensor",
            role="input",
            dims_count=3,
            shape=previous_shape,
            data_type="float_hidden_tensor",
            source=f"layer_{previous_layer_index}_output"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="residual_connection",
            tensor_name="residual_source_tensor",
            role="skip_input",
            dims_count=3,
            shape=previous_shape,
            data_type="float_hidden_tensor",
            source=f"layer_{source_layer_index}_output"
        ))

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="residual_connection",
            weight_name="residual_connection_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="residual_connection",
            tensor_name="output_residual_tensor",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type="float_hidden_tensor",
            source="previous_layer_output_plus_residual_source_tensor"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="residual_connection",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="residual_connection",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_sequence_length"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="residual_connection",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(model_dim),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_model_dim"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="residual_connection",
            layer_kind="merge_layer",
            layer_role="residual_addition",
            repeat_policy="repeat_allowed",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="residual_connection",
            stage_name="residual_connection_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="residual_connection",
            step_key=f"layer_{layer_index}_residual_previous_output_shape",
            previous_step_key="none",
            step_order=1,
            title="Previous layer output shape",
            formula="previous_layer_output_shape",
            substitution=f"Layer {previous_layer_index} output shape = {previous_shape}",
            result=previous_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="residual_connection",
            step_key=f"layer_{layer_index}_residual_source_shape",
            previous_step_key=f"layer_{layer_index}_residual_previous_output_shape",
            step_order=2,
            title="Residual source tensor shape",
            formula="residual_source_shape",
            substitution=f"Layer {source_layer_index} output shape = {previous_shape}",
            result=previous_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="residual_connection",
            step_key=f"layer_{layer_index}_residual_output_shape",
            previous_step_key=f"layer_{layer_index}_residual_source_shape",
            step_order=3,
            title="Residual addition output shape",
            formula="same_shape_tensor + same_shape_tensor keeps the same shape",
            substitution=f"{previous_shape} + {previous_shape}",
            result=output_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="residual_connection",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
