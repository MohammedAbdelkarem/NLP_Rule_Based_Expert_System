from src.facts import CalculationStep, ComponentGroup, Layer, LayerMetadata, TensorInfo, WeightShape


def shape_3d(batch_size, sequence_length, feature_dim):
    return f"({int(batch_size)}, {int(sequence_length)}, {int(feature_dim)})"


def encoder_decoder_weight_summary(input_dim, hidden_size):
    return (
        f"encoder_recurrent_core: ({int(input_dim)}, {int(hidden_size)}); "
        f"decoder_recurrent_core: ({int(hidden_size)}, {int(hidden_size)}); "
        f"context_bridge: ({int(hidden_size)}, {int(hidden_size)})"
    )


def declare_encoder_decoder_group(
    engine,
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    batch_size,
    source_sequence_length,
    target_sequence_length,
    input_dim,
    hidden_size,
    parameter_shape,
):
    engine.declare(ComponentGroup(
        group_index=layer_index,
        group_name="rnn_encoder_decoder",
        group_title=layer_title,
        internal_layers="encoder recurrent layer + context state + decoder recurrent layer",
        combine_operation="decoder conditions on encoded source context",
        input_shape=input_shape,
        output_shape=output_shape,
        status="computed"
    ))
    engine.declare(Layer(
        layer_index=layer_index,
        layer_name="rnn_encoder_decoder",
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape
    ))
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name="rnn_encoder_decoder",
        weight_name="encoder_decoder_recurrent_parameters",
        shape=parameter_shape
    ))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_encoder_decoder", tensor_name="encoder_input_sequence", role="input", dims_count=3, shape=input_shape, data_type="float_source_sequence_tensor", source="source_sequence"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_encoder_decoder", tensor_name="decoder_output_sequence", role="output", dims_count=3, shape=output_shape, data_type="float_decoder_sequence_tensor", source="decoder_recurrent_output"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_encoder_decoder", tensor_name="output_batch_size", role="shape_metadata", dims_count=1, shape=str(int(batch_size)), data_type="integer_dimension_value", source="batch_size"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_encoder_decoder", tensor_name="output_sequence_length", role="shape_metadata", dims_count=1, shape=str(int(target_sequence_length)), data_type="integer_dimension_value", source="target_sequence_length"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_encoder_decoder", tensor_name="source_sequence_length", role="shape_metadata", dims_count=1, shape=str(int(source_sequence_length)), data_type="integer_dimension_value", source="source_sequence_length"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_encoder_decoder", tensor_name="output_model_dim", role="shape_metadata", dims_count=1, shape=str(int(hidden_size)), data_type="integer_dimension_value", source="decoder_hidden_size"))
    engine.declare(TensorInfo(layer_index=layer_index, layer_name="rnn_encoder_decoder", tensor_name="input_model_dim", role="shape_metadata", dims_count=1, shape=str(int(input_dim)), data_type="integer_dimension_value", source="encoder_input_dim"))
    engine.declare(LayerMetadata(layer_index=layer_index, layer_name="rnn_encoder_decoder", layer_kind="sequence_to_sequence_component_group", layer_role="encoder_decoder_generation_architecture", repeat_policy="allowed_with_new_source_target_lengths", consecutive_repeat_policy="not_recommended", terminal_status="non_terminal"))
    engine.declare(CalculationStep(layer_index=layer_index, layer_name="rnn_encoder_decoder", step_key=f"layer_{layer_index}_rnn_encoder_decoder_output", previous_step_key="source_sequence_input", step_order=1, title="RNN encoder-decoder output shape", formula="decoder_output = (batch_size, target_sequence_length, hidden_size)", substitution=output_shape, result=output_shape))
