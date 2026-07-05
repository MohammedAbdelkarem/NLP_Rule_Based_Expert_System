from experta import Rule, MATCH, NOT

from src.facts import ComponentGroup, OutputStatus, TensorInfo, WeightShape
from src.utils.printing import print_skipgram_sgns_result


class SkipGramSGNSOutputRules:
    @Rule(
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="skipgram_sgns",
            group_title=MATCH.layer_title,
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed"
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            tensor_name="target_vocab_size",
            shape=MATCH.target_vocab_size
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            tensor_name="context_vocab_size",
            shape=MATCH.context_vocab_size
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            tensor_name="embedding_dim",
            shape=MATCH.embedding_dim
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            tensor_name="context_window_size",
            shape=MATCH.context_window_size
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            tensor_name="negative_sample_count",
            shape=MATCH.negative_sample_count
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            weight_name="trainable_embedding_matrices",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_skipgram_sgns_component_result(
        self,
        layer_index,
        layer_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape,
        target_vocab_size,
        context_vocab_size,
        embedding_dim,
        context_window_size,
        negative_sample_count,
        parameter_shape
    ):
        print_skipgram_sgns_result(
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            internal_layers=internal_layers,
            combine_operation=combine_operation,
            target_vocab_size=int(target_vocab_size),
            context_vocab_size=int(context_vocab_size),
            embedding_dim=int(embedding_dim),
            context_window_size=int(context_window_size),
            negative_sample_count=int(negative_sample_count)
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
