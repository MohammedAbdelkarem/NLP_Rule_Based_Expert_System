from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import (
    SelectedLayer,
    OptionSelection,
    TensorInfo,
    LayerMetadata,
    RollbackRequest,
    TransformationSuggestion
)


class ActivationCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="activation_input_data_kind",
            value="raw_token_ids",
            status="resolved"
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name
        ),
        TEST(lambda layer_name: layer_name == "relu" or layer_name == "gelu"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name
        )),
        salience=80
    )
    def rollback_activation_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index,
        layer_name
    ):
        self.retract(option_fact)

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name=layer_name,
            reason="raw_token_ids_need_input_encoding_before_activation",
            message=(
                "Activation layers cannot be applied directly to raw integer token IDs. "
                "Choose Input Encoding first, or provide an already encoded float tensor."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.current_layer_name
        ),
        LayerMetadata(
            layer_index=MATCH.previous_layer_index,
            layer_role="activation"
        ),
        TEST(lambda current_layer_name: current_layer_name == "relu" or current_layer_name == "gelu" or current_layer_name == "softmax"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.current_layer_name
        )),
        salience=81
    )
    def rollback_consecutive_activation_layer(
        self,
        layer_index,
        previous_layer_index,
        current_layer_name
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name=current_layer_name,
            reason="consecutive_activation_layer",
            message=(
                "An activation layer was selected directly after another activation layer. "
                "Consecutive activation layers are usually redundant. "
                "Choose a transformation layer before adding another activation."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name
        ),
        TEST(lambda layer_name: layer_name == "relu" or layer_name == "gelu"),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "float_hidden_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name
        )),
        salience=80
    )
    def rollback_activation_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        layer_name,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name=layer_name,
            from_shape=previous_shape,
            to_shape="float_hidden_tensor",
            method="Use Input Encoding or another suitable adapter before the activation layer."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name=layer_name,
            reason="activation_requires_float_hidden_tensor",
            message=(
                "Activation layers require a float hidden tensor. "
                "The previous layer output data type is not compatible."
            )
        ))
