from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter, OptionSelection
from src.prompts import (
    LOG_MEL_SPECTROGRAM_SELECTED,
    ASK_LOG_MEL_EPSILON,
    ASK_LOG_MEL_COMPRESSION_MODE,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class LogMelSpectrogramParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            parameter_name="epsilon_power"
        )),
        salience=70
    )
    def ask_log_mel_spectrogram_parameters(self, layer_index):
        print(LOG_MEL_SPECTROGRAM_SELECTED)

        epsilon_power = read_int_parameter(ASK_LOG_MEL_EPSILON)
        compression_mode_choice = read_text_parameter(ASK_LOG_MEL_COMPRESSION_MODE)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            parameter_name="epsilon_power",
            value=epsilon_power
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice=compression_mode_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.epsilon_power_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            parameter_name="epsilon_power",
            value=MATCH.epsilon_power
        ),
        TEST(lambda epsilon_power: int(epsilon_power) <= 0),
        salience=69
    )
    def ask_epsilon_again_if_not_positive(self, epsilon_power_fact, layer_index, epsilon_power):
        print()
        print("Invalid epsilon exponent. It must be greater than 0 so epsilon is a small positive value.")

        self.retract(epsilon_power_fact)

        new_epsilon_power = read_int_parameter(ASK_LOG_MEL_EPSILON)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            parameter_name="epsilon_power",
            value=new_epsilon_power
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_natural_log_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice="1",
            value="natural_log",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_log10_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice="2",
            value="log10",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice="3",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_decibel_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice="3",
            value="decibel",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        salience=68
    )
    def ask_log_mel_compression_mode_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid Log-Mel compression mode choice. Please choose 1, 2, or 3.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_LOG_MEL_COMPRESSION_MODE)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))
