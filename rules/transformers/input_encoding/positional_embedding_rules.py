from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    LayerParameter,
    IntermediateTensor,
    CalculationStep
)


class InputEncodingPositionalEmbeddingRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="input_encoding"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            tensor_name="token_embedding_output"
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            tensor_name="positional_embedding_output"
        )),
        salience=63
    )
    def calculate_positional_embedding_shapes(
        self,
        layer_index,
        batch_size,
        sequence_length,
        model_dim
    ):
        position_indices_shape = f"({batch_size}, {sequence_length})"
        positional_embedding_matrix_shape = f"({sequence_length}, {model_dim})"
        positional_embedding_output_shape = f"({batch_size}, {sequence_length}, {model_dim})"

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="position_indices",
            shape=position_indices_shape,
            data_type="integer_position_indices",
            source="sequence_positions"
        ))

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="positional_embedding_output",
            shape=positional_embedding_output_shape,
            data_type="float_tensor",
            source="position_indices_lookup_in_positional_embedding_matrix"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="input_encoding",
            step_key=f"layer_{layer_index}_positional_embedding_matrix_shape",
            previous_step_key=f"layer_{layer_index}_token_embedding_output_shape",
            step_order=4,
            title="Positional embedding matrix shape",
            formula="(sequence_length, model_dim)",
            substitution=f"({sequence_length}, {model_dim})",
            result=positional_embedding_matrix_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="input_encoding",
            step_key=f"layer_{layer_index}_positional_embedding_output_shape",
            previous_step_key=f"layer_{layer_index}_positional_embedding_matrix_shape",
            step_order=5,
            title="Positional embedding output shape",
            formula="(batch_size, sequence_length, model_dim)",
            substitution=f"({batch_size}, {sequence_length}, {model_dim})",
            result=positional_embedding_output_shape
        ))
