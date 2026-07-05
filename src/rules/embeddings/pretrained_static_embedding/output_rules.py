from experta import MATCH, NOT, Rule

from src.facts import Layer, OptionSelection, OutputStatus, TensorInfo, WeightShape
from src.utils.print_helpers import print_pretrained_static_embedding_result


class PretrainedStaticEmbeddingOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            weight_name="pretrained_embedding_matrix",
            shape=MATCH.parameter_shape,
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            tensor_name="pretrained_static_embedding_output",
            role="output",
            data_type=MATCH.output_data_type,
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            value=MATCH.source_value,
            status="resolved",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            value=MATCH.trainable_value,
            status="resolved",
        ),
        NOT(
            OutputStatus(
                layer_index=MATCH.layer_index,
                output_name="layer_result_printed",
                status="printed",
            ),
        ),
        salience=50,
    )
    def print_pretrained_static_embedding_layer_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        parameter_shape,
        output_data_type,
        source_value,
        trainable_value
    ):
        print_pretrained_static_embedding_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="pretrained_static_embedding",
            layer_kind="simple_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            output_data_type=output_data_type,
            source_value=source_value,
            trainable_value=trainable_value,
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed",
        ))
