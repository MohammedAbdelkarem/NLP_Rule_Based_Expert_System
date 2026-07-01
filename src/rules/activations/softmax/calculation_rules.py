from experta import Rule, MATCH, NOT, TEST

from src.facts import SelectedLayer, LayerParameter, TensorInfo, Layer
from src.rules.activations.common import declare_probability_activation_layer


class SoftmaxCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="softmax",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            parameter_name="vector_dim",
            value=MATCH.vector_dim
        ),
        TEST(lambda vector_dim: vector_dim > 0),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_softmax_layer(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        vector_dim
    ):
        input_shape = f"({batch_size}, {sequence_length}, {vector_dim})"

        declare_probability_activation_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="softmax",
            layer_title=layer_title,
            input_shape=input_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            vector_dim=vector_dim,
            batch_source="softmax_input_batch_size",
            sequence_source="softmax_input_sequence_length",
            vector_dim_source="softmax_vector_dimension",
            substitution=f"input logits shape {input_shape}; normalize over the last dimension"
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="softmax",
            layer_title=MATCH.layer_title
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.previous_output_shape,
            data_type="logits"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.batch_size
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.sequence_length
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.vector_dim
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_softmax_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        batch_size,
        sequence_length,
        vector_dim
    ):
        declare_probability_activation_layer(
            engine=self,
            layer_index=layer_index,
            layer_name="softmax",
            layer_title=layer_title,
            input_shape=previous_output_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            vector_dim=vector_dim,
            batch_source=f"layer_{previous_layer_index}_output_batch_size",
            sequence_source=f"layer_{previous_layer_index}_output_sequence_length",
            vector_dim_source=f"layer_{previous_layer_index}_output_model_dim",
            substitution=f"Layer {previous_layer_index} logits {previous_output_shape}; normalize over the last dimension"
        )
