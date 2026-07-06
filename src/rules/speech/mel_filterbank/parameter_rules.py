from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter
from src.prompts import (
    MEL_FILTERBANK_SELECTED,
    ASK_NUM_MEL_FILTERS,
    ASK_MEL_MIN_FREQUENCY_HZ,
    ASK_MEL_MAX_FREQUENCY_HZ,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class MelFilterbankParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="num_mel_filters"
        )),
        salience=70
    )
    def ask_mel_filterbank_parameters(self, layer_index):
        print(MEL_FILTERBANK_SELECTED)

        num_mel_filters = read_int_parameter(ASK_NUM_MEL_FILTERS)
        min_frequency_hz = read_int_parameter(ASK_MEL_MIN_FREQUENCY_HZ)
        max_frequency_hz = read_int_parameter(ASK_MEL_MAX_FREQUENCY_HZ)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            parameter_name="num_mel_filters",
            value=num_mel_filters
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            parameter_name="min_frequency_hz",
            value=min_frequency_hz
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            parameter_name="max_frequency_hz",
            value=max_frequency_hz
        ))

    @Rule(
        AS.num_mel_filters_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="num_mel_filters",
            value=MATCH.num_mel_filters
        ),
        TEST(lambda num_mel_filters: int(num_mel_filters) <= 0),
        salience=69
    )
    def ask_num_mel_filters_again_if_not_positive(
        self,
        num_mel_filters_fact,
        layer_index,
        num_mel_filters
    ):
        print()
        print("Invalid number of Mel filters. It must be greater than 0.")

        self.retract(num_mel_filters_fact)

        new_num_mel_filters = read_int_parameter(ASK_NUM_MEL_FILTERS)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            parameter_name="num_mel_filters",
            value=new_num_mel_filters
        ))

    @Rule(
        AS.min_frequency_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="min_frequency_hz",
            value=MATCH.min_frequency_hz
        ),
        TEST(lambda min_frequency_hz: int(min_frequency_hz) < 0),
        salience=69
    )
    def ask_min_frequency_again_if_negative(
        self,
        min_frequency_fact,
        layer_index,
        min_frequency_hz
    ):
        print()
        print("Invalid minimum frequency. It must be greater than or equal to 0 Hz.")

        self.retract(min_frequency_fact)

        new_min_frequency_hz = read_int_parameter(ASK_MEL_MIN_FREQUENCY_HZ)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            parameter_name="min_frequency_hz",
            value=new_min_frequency_hz
        ))
