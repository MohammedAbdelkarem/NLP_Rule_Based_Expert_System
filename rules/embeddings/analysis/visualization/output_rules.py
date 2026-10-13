from experta import Rule, MATCH, NOT

from facts import ComponentGroup, OutputStatus, TensorInfo, WeightShape
from utils.printing import print_embedding_visualization_result


class EmbeddingVisualizationOutputRules:
    @Rule(
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="embedding_visualization",
            group_title=MATCH.layer_title,
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed"
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            tensor_name="num_items",
            shape=MATCH.num_items
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            tensor_name="embedding_dim",
            shape=MATCH.embedding_dim
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            tensor_name="visualization_dim",
            shape=MATCH.visualization_dim
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            weight_name="analysis_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_embedding_visualization_component_result(
        self,
        layer_index,
        layer_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape,
        num_items,
        embedding_dim,
        visualization_dim,
        parameter_shape
    ):
        print_embedding_visualization_result(
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            internal_layers=internal_layers,
            combine_operation=combine_operation,
            num_items=int(num_items),
            embedding_dim=int(embedding_dim),
            visualization_dim=int(visualization_dim)
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
