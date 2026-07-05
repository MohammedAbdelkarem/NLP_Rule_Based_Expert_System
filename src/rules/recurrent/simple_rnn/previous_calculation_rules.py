from experta import MATCH, NOT, Rule

from src.facts import Layer, LayerParameter, OptionSelection, SelectedLayer, TensorInfo
from src.rules.recurrent.common import (
    declare_recurrent_layer,
    rnn_weight_summary,
    shape_2d,
    shape_3d,
)


class SimpleRNNPreviousCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="simple_rnn",
            layer_title=MATCH.layer_title
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="simple_rnn",
            question_key="recurrent_return_mode",
            value="full_sequence",
            status="resolved"
        ),
        Layer(layer_index=MATCH.previous_layer_index, output_shape=MATCH.input_shape),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            shape=MATCH.batch_size
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length",
            shape=MATCH.sequence_length
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            shape=MATCH.input_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="simple_rnn",
            parameter_name="hidden_size",
            value=MATCH.hidden_size
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_next_simple_rnn_full_sequence(
        self,
        layer_index,
        layer_title,
        input_shape,
        batch_size,
        sequence_length,
        input_dim,
        hidden_size
    ):
        output_shape = shape_3d(batch_size, sequence_length, hidden_size)
        parameter_shape = rnn_weight_summary(input_dim, hidden_size)
        declare_recurrent_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="simple_rnn",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            output_dims_count=3,
            batch_size=batch_size,
            sequence_length=sequence_length,
            input_dim=input_dim,
            hidden_size=hidden_size,
            weight_name="simple_rnn_parameters",
            parameter_shape=parameter_shape,
            output_tensor_name="output_hidden_sequence",
            output_data_type="float_hidden_sequence_tensor",
            layer_role="sequence_memory_layer",
            return_mode="full_sequence"
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="simple_rnn",
            layer_title=MATCH.layer_title
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="simple_rnn",
            question_key="recurrent_return_mode",
            value="last_hidden",
            status="resolved"
        ),
        Layer(layer_index=MATCH.previous_layer_index, output_shape=MATCH.input_shape),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            shape=MATCH.batch_size
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length",
            shape=MATCH.sequence_length
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            shape=MATCH.input_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="simple_rnn",
            parameter_name="hidden_size",
            value=MATCH.hidden_size
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_next_simple_rnn_last_hidden(
        self,
        layer_index,
        layer_title,
        input_shape,
        batch_size,
        sequence_length,
        input_dim,
        hidden_size
    ):
        output_shape = shape_2d(batch_size, hidden_size)
        parameter_shape = rnn_weight_summary(input_dim, hidden_size)
        declare_recurrent_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="simple_rnn",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            output_dims_count=2,
            batch_size=batch_size,
            sequence_length=sequence_length,
            input_dim=input_dim,
            hidden_size=hidden_size,
            weight_name="simple_rnn_parameters",
            parameter_shape=parameter_shape,
            output_tensor_name="output_last_hidden_state",
            output_data_type="float_hidden_state_tensor",
            layer_role="sequence_memory_layer",
            return_mode="last_hidden"
        )
