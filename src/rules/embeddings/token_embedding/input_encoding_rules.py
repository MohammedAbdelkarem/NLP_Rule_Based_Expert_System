from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    IntermediateTensor,
    CalculationStep,
)
from src.rules.embeddings.token_embedding.common import (
    build_raw_token_shape,
    build_token_embedding_matrix_shape,
    build_token_embedding_output_shape,
)


class InputEncodingTokenEmbeddingRules:
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
            parameter_name="vocab_size",
            value=MATCH.vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            tensor_name="token_embedding_output"
        )),
        salience=64
    )
    def calculate_input_encoding_token_embedding_shapes(
        self,
        layer_index,
        batch_size,
        sequence_length,
        vocab_size,
        model_dim
    ):
        raw_token_shape = build_raw_token_shape(batch_size, sequence_length)
        matrix_shape = build_token_embedding_matrix_shape(vocab_size, model_dim)
        output_shape = build_token_embedding_output_shape(
            batch_size,
            sequence_length,
            model_dim
        )

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="raw_token_ids",
            role="input",
            dims_count=2,
            shape=raw_token_shape,
            data_type="integer_token_indices",
            source="tokenizer_output"
        ))

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="token_embedding_output",
            shape=output_shape,
            data_type="float_tensor",
            source="shared_token_embedding_lookup"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="input_encoding",
            step_key=f"layer_{layer_index}_raw_token_shape",
            previous_step_key="none",
            step_order=1,
            title="Raw token ID input shape",
            formula="(batch_size, sequence_length)",
            substitution=f"({batch_size}, {sequence_length})",
            result=raw_token_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="input_encoding",
            step_key=f"layer_{layer_index}_token_embedding_matrix_shape",
            previous_step_key=f"layer_{layer_index}_raw_token_shape",
            step_order=2,
            title="Token embedding matrix shape",
            formula="(vocab_size, model_dim)",
            substitution=f"({vocab_size}, {model_dim})",
            result=matrix_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="input_encoding",
            step_key=f"layer_{layer_index}_token_embedding_output_shape",
            previous_step_key=f"layer_{layer_index}_token_embedding_matrix_shape",
            step_order=3,
            title="Token embedding output shape",
            formula="(batch_size, sequence_length, model_dim)",
            substitution=f"({batch_size}, {sequence_length}, {model_dim})",
            result=output_shape
        ))
