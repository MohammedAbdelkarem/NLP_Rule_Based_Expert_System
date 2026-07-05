from experta import Rule, MATCH, NOT

from src.facts import ComponentGroup, LayerParameter, SelectedLayer
from src.rules.embeddings.analysis.common import (
    declare_analysis_component_group,
    declare_analysis_step,
    declare_evaluation_metadata,
    matrix_shape,
)


class EmbeddingEvaluationCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            parameter_name="evaluation_pair_count",
            value=MATCH.evaluation_pair_count
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            parameter_name="benchmark_count",
            value=MATCH.benchmark_count
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_embedding_evaluation_shape(
        self,
        layer_index,
        layer_title,
        evaluation_pair_count,
        embedding_dim,
        benchmark_count
    ):
        input_shape = matrix_shape(evaluation_pair_count, embedding_dim)
        output_shape = matrix_shape(benchmark_count, 1)
        parameter_shape = "no trainable weights; benchmark scoring only"

        declare_analysis_component_group(
            engine=self,
            layer_index=layer_index,
            group_name="embedding_evaluation",
            group_title=layer_title,
            layer_kind="embedding_evaluation_component_group",
            layer_role="evaluate_embedding_quality_with_similarity_or_analogy_tasks",
            internal_layers=(
                "benchmark pair reader + similarity score computation + "
                "evaluation report generator"
            ),
            combine_operation="compare model scores with benchmark targets",
            input_shape=input_shape,
            output_shape=output_shape,
            tensor_name="embedding_evaluation_report",
            tensor_type="embedding_evaluation_score_matrix",
            parameter_name="evaluation_parameters",
            parameter_shape=parameter_shape
        )
        declare_evaluation_metadata(
            engine=self,
            layer_index=layer_index,
            evaluation_pair_count=evaluation_pair_count,
            embedding_dim=embedding_dim,
            benchmark_count=benchmark_count
        )
        declare_analysis_step(
            engine=self,
            layer_index=layer_index,
            layer_name="embedding_evaluation",
            step_title="Create embedding evaluation report",
            output_shape=output_shape
        )
