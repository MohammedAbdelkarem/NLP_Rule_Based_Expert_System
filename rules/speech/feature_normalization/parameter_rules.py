from experta import Rule, MATCH, NOT, AS, TEST

from facts import SelectedLayer, LayerParameter
from prompts import FEATURE_NORMALIZATION_SELECTED, ASK_FEATURE_NORMALIZATION_MODE


from utils.console_control import read_int_parameter, read_text_parameter
class FeatureNormalizationParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization",
            parameter_name="normalization_mode"
        )),
        salience=70
    )
    def ask_feature_normalization_parameters(self, layer_index):
        print(FEATURE_NORMALIZATION_SELECTED)

        normalization_mode = read_int_parameter(ASK_FEATURE_NORMALIZATION_MODE)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feature_normalization",
            parameter_name="normalization_mode",
            value=normalization_mode
        ))

    @Rule(
        AS.mode_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization",
            parameter_name="normalization_mode",
            value=MATCH.normalization_mode
        ),
        TEST(lambda normalization_mode: str(normalization_mode) not in ("1", "2")),
        salience=69
    )
    def ask_normalization_mode_again_if_invalid(
        self,
        mode_fact,
        layer_index,
        normalization_mode
    ):
        print()
        print("Invalid normalization mode. Choose 1 or 2.")

        self.retract(mode_fact)
        new_mode = read_int_parameter(ASK_FEATURE_NORMALIZATION_MODE)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feature_normalization",
            parameter_name="normalization_mode",
            value=new_mode
        ))
