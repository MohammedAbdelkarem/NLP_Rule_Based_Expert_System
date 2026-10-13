from experta import MATCH, NOT, Rule

from facts import OptionSelection, SelectedLayer, TensorInfo, TrainingComponent
from rules.losses.common import declare_training_loss


class BinaryCrossEntropyCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="binary_cross_entropy",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            shape=MATCH.prediction_shape,
            data_type="binary_probability",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            shape=MATCH.batch_size,
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="binary_cross_entropy",
            question_key="loss_reduction",
            value=MATCH.reduction,
            status="resolved",
        ),
        NOT(TrainingComponent(component_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_rank_2_bce(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        batch_size,
        reduction
    ):
        declare_training_loss(
            self,
            layer_index,
            "binary_cross_entropy",
            layer_title,
            "binary_cross_entropy",
            prediction_shape,
            prediction_shape,
            "binary_probability",
            "BCE = -[y log(p) + (1-y) log(1-p)]",
            f"predictions {prediction_shape}; labels {prediction_shape}",
            reduction,
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="binary_cross_entropy",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.prediction_shape,
            data_type="binary_probability",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            shape=MATCH.batch_size,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length",
            role="shape_metadata",
            shape=MATCH.sequence_length,
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="binary_cross_entropy",
            question_key="loss_reduction",
            value=MATCH.reduction,
            status="resolved",
        ),
        NOT(TrainingComponent(component_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_rank_3_bce(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        batch_size,
        sequence_length,
        reduction
    ):
        declare_training_loss(
            self,
            layer_index,
            "binary_cross_entropy",
            layer_title,
            "binary_cross_entropy",
            prediction_shape,
            prediction_shape,
            "binary_probability",
            "BCE = -[y log(p) + (1-y) log(1-p)]",
            f"sequence predictions {prediction_shape}; labels {prediction_shape}",
            reduction,
        )
