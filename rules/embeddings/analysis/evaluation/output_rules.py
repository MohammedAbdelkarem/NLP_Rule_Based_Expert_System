from experta import Rule, MATCH, NOT

from facts import ComponentGroup, OutputStatus, TensorInfo, WeightShape
from utils.printing import print_embedding_evaluation_result


class EmbeddingEvaluationOutputRules:
    @Rule(
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="embedding_evaluation",
            group_title=MATCH.layer_title,
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed"
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            tensor_name="evaluation_pair_count",
            shape=MATCH.evaluation_pair_count
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            tensor_name="embedding_dim",
            shape=MATCH.embedding_dim
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            tensor_name="benchmark_count",
            shape=MATCH.benchmark_count
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            weight_name="evaluation_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_embedding_evaluation_component_result(
        self,
        layer_index,
        layer_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape,
        evaluation_pair_count,
        embedding_dim,
        benchmark_count,
        parameter_shape
    ):
        print_embedding_evaluation_result(
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            internal_layers=internal_layers,
            combine_operation=combine_operation,
            evaluation_pair_count=int(evaluation_pair_count),
            embedding_dim=int(embedding_dim),
            benchmark_count=int(benchmark_count)
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
