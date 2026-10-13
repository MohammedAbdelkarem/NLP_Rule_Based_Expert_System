from experta import MATCH, NOT, Rule, TEST

from facts import OptionSelection, SelectedLayer, TensorInfo, TrainingComponent
from rules.classification_heads.common import (
    make_rank_2_class_index_label_shape,
    make_rank_2_shape,
    make_rank_3_class_index_label_shape,
    make_rank_3_shape,
)
from rules.losses.common import declare_training_loss


class CategoricalCrossEntropyCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="categorical_cross_entropy",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            shape=MATCH.prediction_shape,
            data_type=MATCH.prediction_data_type,
        ),
        TEST(
            lambda prediction_data_type: (
                prediction_data_type == "class_probabilities"
                or prediction_data_type == "probabilities"
            ),
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            shape=MATCH.batch_size,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            shape=MATCH.class_count,
        ),
        TEST(lambda class_count: int(class_count) > 1),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            value="one_hot",
            status="resolved",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="loss_reduction",
            value=MATCH.reduction,
            status="resolved",
        ),
        NOT(TrainingComponent(component_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_rank_2_cce_one_hot(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        batch_size,
        class_count,
        reduction
    ):
        label_shape = make_rank_2_shape(batch_size, class_count)
        self._declare_cce(
            layer_index=layer_index,
            layer_title=layer_title,
            prediction_shape=prediction_shape,
            label_shape=label_shape,
            class_count=class_count,
            reduction=reduction,
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="categorical_cross_entropy",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            shape=MATCH.prediction_shape,
            data_type=MATCH.prediction_data_type,
        ),
        TEST(
            lambda prediction_data_type: (
                prediction_data_type == "class_probabilities"
                or prediction_data_type == "probabilities"
            ),
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            shape=MATCH.batch_size,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            shape=MATCH.class_count,
        ),
        TEST(lambda class_count: int(class_count) > 1),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            value="class_index",
            status="resolved",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="loss_reduction",
            value=MATCH.reduction,
            status="resolved",
        ),
        NOT(TrainingComponent(component_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_rank_2_cce_class_index(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        batch_size,
        class_count,
        reduction
    ):
        label_shape = make_rank_2_class_index_label_shape(batch_size)
        self._declare_cce(
            layer_index=layer_index,
            layer_title=layer_title,
            prediction_shape=prediction_shape,
            label_shape=label_shape,
            class_count=class_count,
            reduction=reduction,
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="categorical_cross_entropy",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.prediction_shape,
            data_type=MATCH.prediction_data_type,
        ),
        TEST(
            lambda prediction_data_type: (
                prediction_data_type == "class_probabilities"
                or prediction_data_type == "probabilities"
            ),
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
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            shape=MATCH.class_count,
        ),
        TEST(lambda class_count: int(class_count) > 1),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            value="one_hot",
            status="resolved",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="loss_reduction",
            value=MATCH.reduction,
            status="resolved",
        ),
        NOT(TrainingComponent(component_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_rank_3_cce_one_hot(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        batch_size,
        sequence_length,
        class_count,
        reduction
    ):
        label_shape = make_rank_3_shape(batch_size, class_count, sequence_length)
        self._declare_sequence_cce(
            layer_index=layer_index,
            layer_title=layer_title,
            prediction_shape=prediction_shape,
            label_shape=label_shape,
            class_count=class_count,
            reduction=reduction,
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="categorical_cross_entropy",
            layer_title=MATCH.layer_title,
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.prediction_shape,
            data_type=MATCH.prediction_data_type,
        ),
        TEST(
            lambda prediction_data_type: (
                prediction_data_type == "class_probabilities"
                or prediction_data_type == "probabilities"
            ),
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
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            shape=MATCH.class_count,
        ),
        TEST(lambda class_count: int(class_count) > 1),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            value="class_index",
            status="resolved",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="loss_reduction",
            value=MATCH.reduction,
            status="resolved",
        ),
        NOT(TrainingComponent(component_index=MATCH.layer_index)),
        salience=60,
    )
    def calculate_rank_3_cce_class_index(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        batch_size,
        sequence_length,
        class_count,
        reduction
    ):
        label_shape = make_rank_3_class_index_label_shape(batch_size, sequence_length)
        self._declare_sequence_cce(
            layer_index=layer_index,
            layer_title=layer_title,
            prediction_shape=prediction_shape,
            label_shape=label_shape,
            class_count=class_count,
            reduction=reduction,
        )

    def _declare_cce(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        label_shape,
        class_count,
        reduction
    ):
        declare_training_loss(
            engine=self,
            layer_index=layer_index,
            layer_name="categorical_cross_entropy",
            layer_title=layer_title,
            loss_type="categorical_cross_entropy",
            input_shape=prediction_shape,
            label_shape=label_shape,
            prediction_data_type="class_probabilities",
            formula="CCE = -Σ y_k log(p_k) = -log(p_correct)",
            substitution=f"predictions {prediction_shape}; labels {label_shape}; classes={class_count}",
            reduction=reduction,
        )

    def _declare_sequence_cce(
        self,
        layer_index,
        layer_title,
        prediction_shape,
        label_shape,
        class_count,
        reduction
    ):
        declare_training_loss(
            engine=self,
            layer_index=layer_index,
            layer_name="categorical_cross_entropy",
            layer_title=layer_title,
            loss_type="categorical_cross_entropy",
            input_shape=prediction_shape,
            label_shape=label_shape,
            prediction_data_type="class_probabilities",
            formula="CCE = -Σ y_k log(p_k) = -log(p_correct)",
            substitution=f"sequence predictions {prediction_shape}; labels {label_shape}; classes={class_count}",
            reduction=reduction,
        )
