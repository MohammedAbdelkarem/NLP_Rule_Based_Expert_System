from experta import Rule, MATCH, NOT

from src.facts import ComponentGroup, LayerParameter, SelectedLayer
from src.rules.embeddings.analysis.common import (
    declare_analysis_component_group,
    declare_analysis_step,
    declare_visualization_metadata,
    matrix_shape,
)


class EmbeddingVisualizationCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            parameter_name="num_items",
            value=MATCH.num_items
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            parameter_name="visualization_dim",
            value=MATCH.visualization_dim
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_embedding_visualization_shape(
        self,
        layer_index,
        layer_title,
        num_items,
        embedding_dim,
        visualization_dim
    ):
        input_shape = matrix_shape(num_items, embedding_dim)
        output_shape = matrix_shape(num_items, visualization_dim)
        parameter_shape = "no trainable weights; projection method is analysis-only"

        declare_analysis_component_group(
            engine=self,
            layer_index=layer_index,
            group_name="embedding_visualization",
            group_title=layer_title,
            layer_kind="embedding_analysis_component_group",
            layer_role="project_embeddings_to_visual_coordinates",
            internal_layers=(
                "embedding matrix reader + dimensionality reduction projection + "
                "coordinate table preparation"
            ),
            combine_operation="convert dense embedding vectors into low-dimensional coordinates",
            input_shape=input_shape,
            output_shape=output_shape,
            tensor_name="embedding_coordinates",
            tensor_type="visualization_coordinate_matrix",
            parameter_name="analysis_parameters",
            parameter_shape=parameter_shape
        )
        declare_visualization_metadata(
            engine=self,
            layer_index=layer_index,
            num_items=num_items,
            embedding_dim=embedding_dim,
            visualization_dim=visualization_dim
        )
        declare_analysis_step(
            engine=self,
            layer_index=layer_index,
            layer_name="embedding_visualization",
            step_title="Create embedding visualization coordinates",
            output_shape=output_shape
        )
