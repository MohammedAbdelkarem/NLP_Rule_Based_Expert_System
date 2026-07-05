from experta import Rule, MATCH, NOT

from src.facts import ComponentGroup, OutputStatus, TensorInfo, WeightShape
from src.utils.printing import print_co_occurrence_matrix_result


class CoOccurrenceMatrixOutputRules:
    @Rule(
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="co_occurrence_matrix",
            group_title=MATCH.layer_title,
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed"
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            tensor_name="vocab_size",
            shape=MATCH.vocab_size
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            tensor_name="context_vocab_size",
            shape=MATCH.context_vocab_size
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            tensor_name="context_window_size",
            shape=MATCH.context_window_size
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="co_occurrence_matrix",
            weight_name="co_occurrence_statistics",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_co_occurrence_matrix_component_result(
        self,
        layer_index,
        layer_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape,
        vocab_size,
        context_vocab_size,
        context_window_size,
        parameter_shape
    ):
        print_co_occurrence_matrix_result(
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            internal_layers=internal_layers,
            combine_operation=combine_operation,
            vocab_size=int(vocab_size),
            context_vocab_size=int(context_vocab_size),
            context_window_size=int(context_window_size)
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
