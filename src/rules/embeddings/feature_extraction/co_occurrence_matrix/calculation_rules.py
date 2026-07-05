from experta import Rule, MATCH, NOT

from src.facts import SelectedLayer, LayerParameter, ComponentGroup
from src.rules.embeddings.feature_extraction.common import (
    declare_cooccurrence_steps,
    declare_feature_component_group,
    declare_feature_shape_metadata,
    matrix_shape,
)


class CoOccurrenceMatrixCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            parameter_name="vocab_size",
            value=MATCH.vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            parameter_name="context_vocab_size",
            value=MATCH.context_vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            parameter_name="context_window_size",
            value=MATCH.context_window_size
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_co_occurrence_matrix_shape(
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
            f"fixed corpus co-occurrence matrix {output_shape}"
        )

        declare_feature_component_group(
            engine=self,
            layer_index=layer_index,
            group_name="co_occurrence_matrix",
            group_title=layer_title,
            layer_kind="corpus_statistics_component_group",
            layer_role="build_word_context_co_occurrence_statistics",
            internal_layers=(
                "context window scanner + co-occurrence counter + "
                "word-context matrix builder"
            ),
            combine_operation="count each word and context pair inside the window",
            input_shape=input_shape,
            output_shape=output_shape,
            tensor_name="co_occurrence_matrix",
            tensor_type="corpus_count_matrix",
            parameter_name="co_occurrence_statistics",
            parameter_shape=parameter_shape
        )
        declare_feature_shape_metadata(
            engine=self,
            layer_index=layer_index,
            layer_name="co_occurrence_matrix",
            vocab_size=vocab_size,
            context_vocab_size=context_vocab_size,
            context_window_size=context_window_size
        )
        declare_cooccurrence_steps(
            engine=self,
            layer_index=layer_index,
            vocab_size=vocab_size,
            context_vocab_size=context_vocab_size,
            output_shape=output_shape
        )
