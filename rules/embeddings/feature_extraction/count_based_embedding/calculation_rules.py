from experta import Rule, MATCH, NOT

from facts import SelectedLayer, LayerParameter, ComponentGroup
from rules.embeddings.feature_extraction.common import (
    declare_count_based_steps,
    declare_feature_component_group,
    declare_feature_shape_metadata,
    matrix_shape,
)


class CountBasedEmbeddingCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            parameter_name="vocab_size",
            value=MATCH.vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_vocab_size",
            value=MATCH.context_vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_window_size",
            value=MATCH.context_window_size
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_count_based_embedding_shape(
        self,
        layer_index,
        layer_title,
        vocab_size,
        context_vocab_size,
        context_window_size
    ):
        input_shape = "running_text_corpus_statistics"
        output_shape = matrix_shape(vocab_size, context_vocab_size)
        parameter_shape = (
            "no trainable weights; "
            f"fixed sparse count matrix {output_shape}"
        )

        declare_feature_component_group(
            engine=self,
            layer_index=layer_index,
            group_name="count_based_embedding",
            group_title=layer_title,
            layer_kind="embedding_feature_extraction_component_group",
            layer_role="build_sparse_word_context_embedding_vectors",
            internal_layers=(
                "context window counting + word-context matrix construction + "
                "sparse embedding vector declaration"
            ),
            combine_operation="word meaning is represented by context count vector",
            input_shape=input_shape,
            output_shape=output_shape,
            tensor_name="count_based_embedding_matrix",
            tensor_type="sparse_count_embedding_matrix",
            parameter_name="count_statistics",
            parameter_shape=parameter_shape
        )
        declare_feature_shape_metadata(
            engine=self,
            layer_index=layer_index,
            layer_name="count_based_embedding",
            vocab_size=vocab_size,
            context_vocab_size=context_vocab_size,
            context_window_size=context_window_size
        )
        declare_count_based_steps(
            engine=self,
            layer_index=layer_index,
            vocab_size=vocab_size,
            context_vocab_size=context_vocab_size,
            output_shape=output_shape
        )
