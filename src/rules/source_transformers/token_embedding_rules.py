from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorShape,
    WeightShape,
    Layer,
    LayerExplanation,
    OutputStatus,
    CalculationStep
)

from src.prompts import (
    TOKEN_EMBEDDING_SELECTED,
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_VOCAB_SIZE,
    ASK_EMBEDDING_DIM
)


class TokenEmbeddingRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_token_embedding_information(self, layer_index):
        print(TOKEN_EMBEDDING_SELECTED)

        batch_size = int(input(ASK_BATCH_SIZE))
        sequence_length = int(input(ASK_SEQUENCE_LENGTH))
        vocab_size = int(input(ASK_VOCAB_SIZE))
        embedding_dim = int(input(ASK_EMBEDDING_DIM))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding",
            parameter_name="batch_size",
            value=batch_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding",
            parameter_name="sequence_length",
            value=sequence_length
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding",
            parameter_name="vocab_size",
            value=vocab_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding",
            parameter_name="embedding_dim",
            value=embedding_dim
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            parameter_name="vocab_size",
            value=MATCH.vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_token_embedding_shapes(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        vocab_size,
        embedding_dim
    ):
        input_shape = f"({batch_size}, {sequence_length})"
        weight_shape = f"({vocab_size}, {embedding_dim})"
        output_shape = f"({batch_size}, {sequence_length}, {embedding_dim})"

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="embedding",
            step_key=f"layer_{layer_index}_embedding_input_shape",
            previous_step_key="none",
            step_order=1,
            title="Embedding input shape",
            formula="(batch_size, sequence_length)",
            substitution=f"({batch_size}, {sequence_length})",
            result=input_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="embedding",
            step_key=f"layer_{layer_index}_embedding_weight_shape",
            previous_step_key=f"layer_{layer_index}_embedding_input_shape",
            step_order=2,
            title="Embedding weight matrix shape",
            formula="(vocab_size, embedding_dim)",
            substitution=f"({vocab_size}, {embedding_dim})",
            result=weight_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="embedding",
            step_key=f"layer_{layer_index}_embedding_output_shape",
            previous_step_key=f"layer_{layer_index}_embedding_weight_shape",
            step_order=3,
            title="Embedding output shape",
            formula="(batch_size, sequence_length, embedding_dim)",
            substitution=f"({batch_size}, {sequence_length}, {embedding_dim})",
            result=output_shape
        ))

        self.declare(TensorShape(
            layer_index=layer_index,
            layer_name="embedding",
            shape_role="input",
            dims_count=2,
            shape=input_shape
        ))

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="embedding",
            weight_name="embedding_matrix",
            shape=weight_shape
        ))

        self.declare(TensorShape(
            layer_index=layer_index,
            layer_name="embedding",
            shape_role="output",
            dims_count=3,
            shape=output_shape
        ))

        self.declare(LayerExplanation(
            layer_index=layer_index,
            layer_name="embedding",
            text="Embedding transforms token indices into dense vectors."
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="embedding",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))

    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="embedding",
            weight_name="embedding_matrix",
            shape=MATCH.weight_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="embedding_result",
            status="printed"
        )),
        salience=50
    )
    def print_token_embedding_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        weight_shape,
        output_shape
    ):
        print()
        print(f"Layer {layer_index}: {layer_title}")
        print()
        print("Input shape:")
        print(input_shape)
        print()
        print("Weight matrix shape:")
        print(weight_shape)
        print()
        print("Output shape:")
        print(output_shape)
        print()
        print("Stored Facts:")
        print(f"Layer index: {layer_index}")
        print("Layer name: embedding")
        print(f"Input shape: {input_shape}")
        print(f"Weight matrix shape: {weight_shape}")
        print(f"Output shape: {output_shape}")
        print()

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="embedding_result",
            status="printed"
        ))
