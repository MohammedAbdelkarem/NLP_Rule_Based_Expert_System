from experta import Rule, MATCH, NOT

from src.facts import ComponentGroup, LayerParameter, SelectedLayer
from src.rules.embeddings.training.common import (
    declare_sgns_component,
    declare_sgns_steps,
    declare_training_shape_metadata,
    embedding_matrix_shape,
)


class SkipGramSGNSCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            parameter_name="target_vocab_size",
            value=MATCH.target_vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            parameter_name="context_vocab_size",
            value=MATCH.context_vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            parameter_name="context_window_size",
            value=MATCH.context_window_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            parameter_name="negative_sample_count",
            value=MATCH.negative_sample_count
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_skipgram_sgns_shapes(
        self,
        layer_index,
        layer_title,
        target_vocab_size,
        context_vocab_size,
        embedding_dim,
        context_window_size,
        negative_sample_count
    ):
        input_shape = "target_context_pairs_with_negative_samples"
        output_shape = embedding_matrix_shape(target_vocab_size, embedding_dim)
        context_shape = embedding_matrix_shape(context_vocab_size, embedding_dim)
        parameter_shape = (
            f"E_target={output_shape}; "
            f"E_context={context_shape}; trainable=True"
        )

        declare_sgns_component(
            engine=self,
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape
        )
        declare_training_shape_metadata(
            engine=self,
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            target_vocab_size=target_vocab_size,
            context_vocab_size=context_vocab_size,
            embedding_dim=embedding_dim,
            context_window_size=context_window_size,
            negative_sample_count=negative_sample_count
        )
        declare_sgns_steps(
            engine=self,
            layer_index=layer_index,
            target_vocab_size=target_vocab_size,
            context_vocab_size=context_vocab_size,
            embedding_dim=embedding_dim,
            output_shape=output_shape
        )
