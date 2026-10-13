from experta import Rule, MATCH, NOT

from facts import Layer, OutputStatus, TensorInfo, WeightShape
from utils.printing import print_cosine_similarity_result


class CosineSimilarityOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity",
            tensor_name="vector_dim",
            shape=MATCH.vector_dim
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity",
            weight_name="learnable_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_cosine_similarity_component_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        vector_dim,
        parameter_shape
    ):
        print_cosine_similarity_result(
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            vector_dim=int(vector_dim)
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
