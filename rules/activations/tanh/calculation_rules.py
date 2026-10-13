from experta import MATCH, NOT, Rule

from facts import Layer, LayerParameter, SelectedLayer, TensorInfo
from rules.activations.common import (
    declare_activation_rank_2_layer,
    declare_activation_rank_3_layer,
)


class TanhCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="tanh",
            layer_title=MATCH.layer_title,
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            parameter_name="batch_size",
            value=MATCH.batch_size,
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            parameter_name="sequence_length",
            value=MATCH.sequence_length,
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            parameter_name="model_dim",
            value=MATCH.model_dim,
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_first_tanh_layer(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        model_dim,
    ):
        input_shape = f"({batch_size}, {sequence_length}, {model_dim})"
        declare_activation_rank_3_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="tanh",
            layer_title=layer_title,
            input_shape=input_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_dim=model_dim,
            batch_source="tanh_input_batch_size",
            sequence_source="tanh_input_sequence_length",
            model_dim_source="tanh_input_model_dim",
            formula="tanh(z) = (e^z - e^-z) / (e^z + e^-z); shape is preserved",
            substitution=input_shape,
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="tanh",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            shape=MATCH.previous_output_shape,
            data_type="float_hidden_tensor",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.batch_size,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.model_dim,
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_rank_2_tanh_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        batch_size,
        model_dim,
    ):
        declare_activation_rank_2_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="tanh",
            layer_title=layer_title,
            input_shape=previous_output_shape,
            batch_size=batch_size,
            model_dim=model_dim,
            batch_source=f"layer_{previous_layer_index}_output_batch_size",
            model_dim_source=f"layer_{previous_layer_index}_output_model_dim",
            formula="tanh(z) maps values to [-1, 1]; previous shape is preserved",
            substitution=f"Layer {previous_layer_index} output shape = {previous_output_shape}",
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="tanh",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.previous_output_shape,
            data_type="float_hidden_tensor",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.batch_size,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.sequence_length,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.model_dim,
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_rank_3_tanh_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        batch_size,
        sequence_length,
        model_dim,
    ):
        declare_activation_rank_3_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="tanh",
            layer_title=layer_title,
            input_shape=previous_output_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_dim=model_dim,
            batch_source=f"layer_{previous_layer_index}_output_batch_size",
            sequence_source=f"layer_{previous_layer_index}_output_sequence_length",
            model_dim_source=f"layer_{previous_layer_index}_output_model_dim",
            formula="tanh(z) maps values to [-1, 1]; previous shape is preserved",
            substitution=f"Layer {previous_layer_index} output shape = {previous_output_shape}",
        )
