from experta import MATCH, NOT, Rule, TEST

from src.facts import (
    Layer,
    LayerMetadata,
    LayerParameter,
    OptionSelection,
    SelectedLayer,
    TransformerStage,
)
from src.rules.embeddings.pretrained_static_embedding.common import (
    build_pretrained_embedding_parameter_shape,
    declare_pretrained_embedding_calculation_steps,
    declare_pretrained_embedding_shape_metadata,
    declare_pretrained_embedding_tensors,
)
from src.rules.embeddings.token_embedding.common import (
    build_raw_token_shape,
    build_token_embedding_matrix_shape,
    build_token_embedding_output_shape,
)


class PretrainedStaticEmbeddingCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="pretrained_static_embedding",
            layer_title=MATCH.layer_title,
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="batch_size",
            value=MATCH.batch_size,
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="sequence_length",
            value=MATCH.sequence_length,
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="vocab_size",
            value=MATCH.vocab_size,
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim,
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            value=MATCH.source_value,
            status="resolved",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            value=MATCH.trainable_value,
            status="resolved",
        ),
        TEST(lambda batch_size: batch_size > 0),
        TEST(lambda sequence_length: sequence_length > 0),
        TEST(lambda vocab_size: vocab_size > 0),
        TEST(lambda embedding_dim: embedding_dim > 0),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_pretrained_static_embedding_layer(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        vocab_size,
        embedding_dim,
        source_value,
        trainable_value
    ):
        raw_token_shape = build_raw_token_shape(batch_size, sequence_length)
        matrix_shape = build_token_embedding_matrix_shape(vocab_size, embedding_dim)
        output_shape = build_token_embedding_output_shape(
            batch_size,
            sequence_length,
            embedding_dim
        )
        parameter_shape = build_pretrained_embedding_parameter_shape(
            vocab_size,
            embedding_dim,
            source_value,
            trainable_value
        )

        declare_pretrained_embedding_tensors(
            engine=self,
            layer_index=layer_index,
            raw_token_shape=raw_token_shape,
            matrix_shape=matrix_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            source_value=source_value
        )

        declare_pretrained_embedding_shape_metadata(
            engine=self,
            layer_index=layer_index,
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            source_value=source_value,
            trainable_value=trainable_value
        )

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            layer_kind="simple_layer",
            layer_role="embedding_lookup",
            repeat_policy="once_first_only",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal",
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            stage_name="pretrained_static_embedding_completed",
            status="computed",
        ))

        declare_pretrained_embedding_calculation_steps(
            engine=self,
            layer_index=layer_index,
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            source_value=source_value,
            trainable_value=trainable_value,
            raw_token_shape=raw_token_shape,
            matrix_shape=matrix_shape,
            output_shape=output_shape
        )

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            layer_title=layer_title,
            input_shape=raw_token_shape,
            output_shape=output_shape,
        ))
