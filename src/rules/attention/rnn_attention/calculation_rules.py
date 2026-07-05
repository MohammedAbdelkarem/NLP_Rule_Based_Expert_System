from experta import MATCH, NOT, Rule

from src.facts import ComponentGroup, Layer, LayerMetadata, LayerParameter, OptionSelection, SelectedLayer, TensorInfo, WeightShape


def shape_3d(batch_size, dim_a, dim_b):
    return f"({int(batch_size)}, {int(dim_a)}, {int(dim_b)})"


def bilinear_weight_shape(decoder_hidden_size, encoder_hidden_size):
    return f"W_s: ({int(decoder_hidden_size)}, {int(encoder_hidden_size)})"


def dot_weight_shape():
    return "no trainable scoring matrix; dot(decoder_state, encoder_state)"


def declare_attention_group(engine, layer_index, layer_title, scoring_type, batch_size, source_sequence_length, target_sequence_length, encoder_hidden_size, decoder_hidden_size, parameter_shape):
    weights_shape = shape_3d(batch_size, target_sequence_length, source_sequence_length)
    context_shape = shape_3d(batch_size, target_sequence_length, encoder_hidden_size)
    engine.declare(ComponentGroup(group_index=layer_index, group_name="rnn_attention", group_title=layer_title, internal_layers="attention scoring + softmax alignment + weighted context sum", combine_operation=f"{scoring_type} attention over encoder states", input_shape="encoder_states + decoder_states", output_shape=context_shape, status="computed"))
    engine.declare(Layer(layer_index=layer_index, layer_name="rnn_attention", layer_title=layer_title, input_shape="encoder_states + decoder_states", output_shape=context_shape))
    engine.declare(WeightShape(layer_index=layer_index, layer_name="rnn_attention", weight_name="attention_scoring_parameters", shape=parameter_shape))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_attention", tensor_name="attention_weights", role="intermediate", dims_count=3, shape=weights_shape, data_type="alignment_probability_tensor", source="attention_softmax"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_attention", tensor_name="context_vectors", role="output", dims_count=3, shape=context_shape, data_type="float_context_tensor", source="weighted_encoder_state_sum"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_attention", tensor_name="output_batch_size", role="shape_metadata", dims_count=1, shape=str(int(batch_size)), data_type="integer_dimension_value", source="batch_size"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_attention", tensor_name="output_sequence_length", role="shape_metadata", dims_count=1, shape=str(int(target_sequence_length)), data_type="integer_dimension_value", source="target_sequence_length"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_attention", tensor_name="source_sequence_length", role="shape_metadata", dims_count=1, shape=str(int(source_sequence_length)), data_type="integer_dimension_value", source="source_sequence_length"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_attention", tensor_name="output_model_dim", role="shape_metadata", dims_count=1, shape=str(int(encoder_hidden_size)), data_type="integer_dimension_value", source="encoder_hidden_size"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_attention", tensor_name="decoder_hidden_size", role="shape_metadata", dims_count=1, shape=str(int(decoder_hidden_size)), data_type="integer_dimension_value", source="decoder_hidden_size"))
    engine.declare(LayerMetadata(layer_index=layer_index, layer_name="rnn_attention", layer_kind="attention_component", layer_role="encoder_decoder_context_selection", repeat_policy="allowed_with_new_scoring_type", consecutive_repeat_policy="not_recommended", terminal_status="non_terminal"))


class RNNAttentionCalculationRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="rnn_attention", layer_title=MATCH.layer_title),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", value="dot_product", status="resolved"),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_hidden_match", value="match", status="resolved"),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="batch_size", value=MATCH.batch_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="source_sequence_length", value=MATCH.source_sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="target_sequence_length", value=MATCH.target_sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="encoder_hidden_size", value=MATCH.encoder_hidden_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="decoder_hidden_size", value=MATCH.decoder_hidden_size),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_dot_product_attention(self, layer_index, layer_title, batch_size, source_sequence_length, target_sequence_length, encoder_hidden_size, decoder_hidden_size):
        declare_attention_group(self, layer_index, layer_title, "dot_product", batch_size, source_sequence_length, target_sequence_length, encoder_hidden_size, decoder_hidden_size, dot_weight_shape())

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="rnn_attention", layer_title=MATCH.layer_title),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", value="bilinear", status="resolved"),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="batch_size", value=MATCH.batch_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="source_sequence_length", value=MATCH.source_sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="target_sequence_length", value=MATCH.target_sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="encoder_hidden_size", value=MATCH.encoder_hidden_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="decoder_hidden_size", value=MATCH.decoder_hidden_size),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_bilinear_attention(self, layer_index, layer_title, batch_size, source_sequence_length, target_sequence_length, encoder_hidden_size, decoder_hidden_size):
        parameter_shape = bilinear_weight_shape(decoder_hidden_size, encoder_hidden_size)
        declare_attention_group(self, layer_index, layer_title, "bilinear", batch_size, source_sequence_length, target_sequence_length, encoder_hidden_size, decoder_hidden_size, parameter_shape)
