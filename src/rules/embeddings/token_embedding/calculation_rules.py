from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    IntermediateTensor,
    WeightShape,
    Layer,
    LayerMetadata,
    TransformerStage,
    CalculationStep,
)
from src.rules.embeddings.token_embedding.common import (
    build_raw_token_shape,
    build_token_embedding_matrix_shape,
    build_token_embedding_output_shape,
    build_token_embedding_parameter_shape,
)


class TokenEmbeddingCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="token_embedding",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding",
            parameter_name="vocab_size",
            value=MATCH.vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_token_embedding_layer(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        vocab_size,
        embedding_dim
    ):
        raw_token_shape = build_raw_token_shape(batch_size, sequence_length)
        matrix_shape = build_token_embedding_matrix_shape(vocab_size, embedding_dim)
        output_shape = build_token_embedding_output_shape(
            batch_size,
            sequence_length,
            embedding_dim
        )
        parameter_shape = build_token_embedding_parameter_shape(vocab_size, embedding_dim)

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="token_embedding",
            weight_name="token_embedding_matrix",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="token_embedding",
            tensor_name="raw_token_ids",
            role="input",
            dims_count=2,
            shape=raw_token_shape,
            data_type="integer_token_indices",
            source="tokenizer_output"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="token_embedding",
            tensor_name="token_embedding_output",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type="float_hidden_tensor",
            source="raw_token_ids_lookup_in_token_embedding_matrix"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="token_embedding",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source="batch_size_parameter"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="token_embedding",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source="sequence_length_parameter"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="token_embedding",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(embedding_dim),
            data_type="integer_dimension_value",
            source="embedding_dim_parameter"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="token_embedding",
            layer_kind="simple_layer",
            layer_role="embedding_lookup",
            repeat_policy="once_first_only",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="token_embedding",
            stage_name="token_embedding_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="token_embedding",
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
            layer_name="token_embedding",
            step_key=f"layer_{layer_index}_token_embedding_matrix_shape",
            previous_step_key=f"layer_{layer_index}_raw_token_shape",
            step_order=2,
            title="Token embedding matrix shape",
            formula="(vocab_size, embedding_dim)",
            substitution=f"({vocab_size}, {embedding_dim})",
            result=matrix_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="token_embedding",
            step_key=f"layer_{layer_index}_token_embedding_output_shape",
            previous_step_key=f"layer_{layer_index}_token_embedding_matrix_shape",
            step_order=3,
            title="Token embedding output shape",
            formula="(batch_size, sequence_length, embedding_dim)",
            substitution=f"({batch_size}, {sequence_length}, {embedding_dim})",
            result=output_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="token_embedding",
            layer_title=layer_title,
            input_shape=raw_token_shape,
            output_shape=output_shape
        ))

