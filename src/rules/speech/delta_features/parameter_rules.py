from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter
from src.prompts import DELTA_FEATURES_SELECTED, ASK_DELTA_WINDOW


from src.utils.console_control import read_int_parameter, read_text_parameter
class DeltaFeatureParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="delta_features"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="delta_features",
            parameter_name="delta_window"
        )),
        salience=70
    )
    def ask_delta_feature_parameters(self, layer_index):
        print(DELTA_FEATURES_SELECTED)

        delta_window = read_int_parameter(ASK_DELTA_WINDOW)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="delta_features",
            parameter_name="delta_window",
            value=delta_window
        ))

    @Rule(
        AS.delta_window_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="delta_features",
            parameter_name="delta_window",
            value=MATCH.delta_window
        ),
        TEST(lambda delta_window: int(delta_window) <= 0),
        salience=69
    )
    def ask_delta_window_again_if_not_positive(
        self,
        delta_window_fact,
        layer_index,
        delta_window
    ):
        print()
        print("Invalid delta window. It must be greater than 0.")

        self.retract(delta_window_fact)
        new_delta_window = read_int_parameter(ASK_DELTA_WINDOW)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="delta_features",
            parameter_name="delta_window",
            value=new_delta_window
        ))
