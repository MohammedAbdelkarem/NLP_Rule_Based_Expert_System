from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    Layer,
    LayerMetadata,
    TensorInfo,
    WeightShape,
    CalculationStep,
    OptionSelection,
    TransformerStage
)


class LayerNormFirstCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="layer_norm",
            layer_title=MATCH.layer_title
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            value="encoded_float_tensor",
            status="resolved"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_layer_norm_shapes(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        model_dim
    ):
        input_shape = f"({batch_size}, {sequence_length}, {model_dim})"
        parameter_shape = f"gamma: ({model_dim},), beta: ({model_dim},)"
        output_shape = input_shape

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="input_encoded_tensor",
            role="input",
            dims_count=3,
            shape=input_shape,
            data_type="float_hidden_tensor",
            source="user_provided_encoded_tensor"
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
            source="batch_size_parameter"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source="sequence_length_parameter"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="layer_norm",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(model_dim),
            data_type="integer_dimension_value",
            source="model_dim_parameter"
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
            formula="(batch_size, sequence_length, model_dim)",
            substitution=f"({batch_size}, {sequence_length}, {model_dim})",
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

