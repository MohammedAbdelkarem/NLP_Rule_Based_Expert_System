from experta import Rule, MATCH, NOT

from facts import Layer, LayerParameter, SelectedLayer
from rules.embeddings.similarity.common import (
    declare_similarity_component,
    vector_shape,
)


class CosineSimilarityCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity",
            parameter_name="vector_dim",
            value=MATCH.vector_dim
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_cosine_similarity_shape(self, layer_index, layer_title, vector_dim):
        vector_shape_value = vector_shape(vector_dim)
        input_shape = f"two embedding vectors: {vector_shape_value}, {vector_shape_value}"
        output_shape = "(1,)"
        parameter_shape = "no trainable weights; normalized dot product only"

        declare_similarity_component(
            engine=self,
            layer_index=layer_index,
            layer_title=layer_title,
            vector_dim=vector_dim,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape
        )
