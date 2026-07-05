from experta import Rule, MATCH, NOT

from src.facts import ComponentGroup, LayerParameter, SelectedLayer
from src.rules.embeddings.analysis.common import (
    declare_analysis_component_group,
    declare_analysis_step,
    declare_bias_metadata,
    matrix_shape,
)


class EmbeddingBiasAnalysisCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding_bias_analysis",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name="target_word_count",
            value=MATCH.target_word_count
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name="attribute_word_count",
            value=MATCH.attribute_word_count
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_embedding_bias_analysis_shape(
        self,
        layer_index,
        layer_title,
        target_word_count,
        attribute_word_count,
        embedding_dim
    ):
        input_shape = (
            f"targets={matrix_shape(target_word_count, embedding_dim)}; "
            f"attributes={matrix_shape(attribute_word_count, embedding_dim)}"
        )
        output_shape = matrix_shape(target_word_count, attribute_word_count)
        parameter_shape = "no trainable weights; cosine association statistics only"

        declare_analysis_component_group(
            engine=self,
            layer_index=layer_index,
            group_name="embedding_bias_analysis",
            group_title=layer_title,
            layer_kind="embedding_bias_analysis_component_group",
            layer_role="inspect_embedding_association_patterns",
            internal_layers=(
                "target vector selector + attribute vector selector + "
                "cosine association matrix + bias report"
            ),
            combine_operation="compare target vectors against attribute vectors",
            input_shape=input_shape,
            output_shape=output_shape,
            tensor_name="bias_association_matrix",
            tensor_type="embedding_bias_score_matrix",
            parameter_name="analysis_parameters",
            parameter_shape=parameter_shape
        )
        declare_bias_metadata(
            engine=self,
            layer_index=layer_index,
            target_word_count=target_word_count,
            attribute_word_count=attribute_word_count,
            embedding_dim=embedding_dim
        )
        declare_analysis_step(
            engine=self,
            layer_index=layer_index,
            layer_name="embedding_bias_analysis",
            step_title="Create embedding bias association scores",
            output_shape=output_shape
        )
