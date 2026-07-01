from experta import Rule, MATCH, NOT

from src.facts import SelectedLayer, LayerParameter, TensorInfo, Layer
from src.rules.activations.common import declare_hidden_activation_layer


class GELUCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="gelu",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="gelu",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="gelu",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="gelu",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_gelu_layer(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        model_dim
    ):
        input_shape = f"({batch_size}, {sequence_length}, {model_dim})"

        declare_hidden_activation_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="gelu",
            layer_title=layer_title,
            dims_count=3,
            input_shape=input_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_dim=model_dim,
            batch_source="gelu_input_batch_size",
            sequence_source="gelu_input_sequence_length",
            model_dim_source="gelu_input_model_dim",
            formula="GELU activation preserves input shape",
            substitution=input_shape
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="gelu",
            layer_title=MATCH.layer_title
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_output_shape,
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
    def calculate_gelu_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        dims_count,
        previous_output_shape,
        batch_size,
        sequence_length,
        model_dim
    ):
        declare_hidden_activation_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="gelu",
            layer_title=layer_title,
            dims_count=dims_count,
            input_shape=previous_output_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_dim=model_dim,
            batch_source=f"layer_{previous_layer_index}_output_batch_size",
            sequence_source=f"layer_{previous_layer_index}_output_sequence_length",
            model_dim_source=f"layer_{previous_layer_index}_output_model_dim",
            formula="GELU activation preserves previous layer output shape",
            substitution=f"Layer {previous_layer_index} output shape = {previous_output_shape}"
        )
