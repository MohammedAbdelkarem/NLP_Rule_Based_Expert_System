from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    Layer,
    LayerMetadata,
    TensorInfo,
    WeightShape,
    CalculationStep,
    TransformerStage
)


class LayerNormPreviousCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="layer_norm",
            layer_title=MATCH.layer_title
        ),
        Layer(
            layer_index=MATCH.previous_layer_index,
            output_shape=MATCH.previous_output_shape
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
    def calculate_layer_norm_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        batch_size,
        sequence_length,
        model_dim
    ):
        input_shape = previous_output_shape
        parameter_shape = f"gamma: ({model_dim},), beta: ({model_dim},)"
        output_shape = previous_output_shape

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="input_hidden_tensor",
            role="input",
            dims_count=3,
            shape=input_shape,
            data_type="float_hidden_tensor",
            source=f"layer_{previous_layer_index}_output"
        ))

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="layer_norm",
            weight_name="layer_norm_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="output_normalized_tensor",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type="float_hidden_tensor",
            source="layer_norm_gamma_beta_transformation"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_sequence_length"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(model_dim),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_output_model_dim"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="layer_norm",
            layer_kind="simple_layer",
            layer_role="normalization",
            repeat_policy="repeat_allowed",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="layer_norm",
            stage_name="normalization_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="layer_norm",
            step_key=f"layer_{layer_index}_layer_norm_input_shape",
            previous_step_key="none",
            step_order=1,
            title="LayerNorm input shape",
            formula="previous_layer_output_shape",
            substitution=f"Layer {previous_layer_index} output shape = {previous_output_shape}",
            result=input_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="layer_norm",
            step_key=f"layer_{layer_index}_layer_norm_parameter_shape",
            previous_step_key=f"layer_{layer_index}_layer_norm_input_shape",
            step_order=2,
            title="LayerNorm learnable parameter shapes",
            formula="gamma shape = (model_dim,), beta shape = (model_dim,)",
            substitution=f"gamma: ({model_dim},), beta: ({model_dim},)",
            result=parameter_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="layer_norm",
            step_key=f"layer_{layer_index}_layer_norm_output_shape",
            previous_step_key=f"layer_{layer_index}_layer_norm_parameter_shape",
            step_order=3,
            title="LayerNorm output shape",
            formula="LayerNorm preserves input shape",
            substitution=input_shape,
            result=output_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="layer_norm",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
