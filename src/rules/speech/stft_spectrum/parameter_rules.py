from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter, OptionSelection
from src.prompts import (
    STFT_SPECTRUM_SELECTED,
    ASK_STFT_N_FFT,
    ASK_STFT_OUTPUT_MODE,
)


class STFTSpectrumParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            parameter_name="n_fft"
        )),
        salience=70
    )
    def ask_stft_spectrum_parameters(self, layer_index):
        print(STFT_SPECTRUM_SELECTED)

        n_fft = int(input(ASK_STFT_N_FFT))
        output_mode_choice = input(ASK_STFT_OUTPUT_MODE)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            parameter_name="n_fft",
            value=n_fft
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice=output_mode_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_magnitude_output_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice="1",
            value="magnitude",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_power_output_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice="2",
            value="power",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice="3",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_complex_output_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice="3",
            value="complex",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        salience=68
    )
    def ask_stft_output_mode_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid STFT output mode choice. Please choose 1, 2, or 3.")

        self.retract(option_fact)

        new_choice = input(ASK_STFT_OUTPUT_MODE)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))
