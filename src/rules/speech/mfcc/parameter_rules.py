from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter, OptionSelection
from src.prompts import (
    MFCC_SELECTED,
    ASK_NUM_MFCC_COEFFICIENTS,
    ASK_MFCC_DCT_TYPE,
    ASK_MFCC_INCLUDE_C0,
    ASK_MFCC_LIFTER,
)


class MFCCParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="mfcc"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            parameter_name="num_mfcc_coefficients"
        )),
        salience=70
    )
    def ask_mfcc_parameters(self, layer_index):
        print(MFCC_SELECTED)

        num_mfcc_coefficients = int(input(ASK_NUM_MFCC_COEFFICIENTS))
        dct_type_choice = input(ASK_MFCC_DCT_TYPE)
        include_c0_choice = input(ASK_MFCC_INCLUDE_C0)
        lifter = int(input(ASK_MFCC_LIFTER))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mfcc",
            parameter_name="num_mfcc_coefficients",
            value=num_mfcc_coefficients
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            choice=dct_type_choice,
            value="unknown",
            status="raw"
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            choice=include_c0_choice,
            value="unknown",
            status="raw"
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mfcc",
            parameter_name="lifter",
            value=lifter
        ))

    @Rule(
        AS.num_mfcc_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            parameter_name="num_mfcc_coefficients",
            value=MATCH.num_mfcc_coefficients
        ),
        TEST(lambda num_mfcc_coefficients: int(num_mfcc_coefficients) <= 0),
        salience=69
    )
    def ask_num_mfcc_again_if_not_positive(
        self,
        num_mfcc_fact,
        layer_index,
        num_mfcc_coefficients
    ):
        print()
        print("Invalid number of MFCC coefficients. It must be greater than 0.")

        self.retract(num_mfcc_fact)

        new_num_mfcc_coefficients = int(input(ASK_NUM_MFCC_COEFFICIENTS))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mfcc",
            parameter_name="num_mfcc_coefficients",
            value=new_num_mfcc_coefficients
        ))

    @Rule(
        AS.lifter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            parameter_name="lifter",
            value=MATCH.lifter
        ),
        TEST(lambda lifter: int(lifter) < 0),
        salience=69
    )
    def ask_lifter_again_if_negative(self, lifter_fact, layer_index, lifter):
        print()
        print("Invalid MFCC lifter value. It must be 0 or greater. Use 0 to disable liftering.")

        self.retract(lifter_fact)

        new_lifter = int(input(ASK_MFCC_LIFTER))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mfcc",
            parameter_name="lifter",
            value=new_lifter
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_dct_type_2(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            choice="1",
            value="dct_2",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_dct_type_3(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            choice="2",
            value="dct_3",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=68
    )
    def ask_mfcc_dct_type_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid MFCC DCT type choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = input(ASK_MFCC_DCT_TYPE)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_include_c0_yes(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            choice="1",
            value="include_c0",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_include_c0_no(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            choice="2",
            value="exclude_c0",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=68
    )
    def ask_mfcc_include_c0_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid include-c0 choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = input(ASK_MFCC_INCLUDE_C0)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))
