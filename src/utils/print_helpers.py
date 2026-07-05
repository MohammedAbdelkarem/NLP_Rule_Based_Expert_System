def print_section_title(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_label_value(label, value):
    print(f"{label}:")
    print(value)


def print_layer_header(layer_index, layer_title):
    print_section_title(f"Layer {layer_index}: {layer_title}")


def print_shape_block(title, shape, data_type):
    print()
    print_label_value(title, shape)
    print(f"Data type: {data_type}")


def print_shape_block_without_type(title, shape):
    print()
    print_label_value(title, shape)


def print_weight_block(parameter_shape):
    print()
    print("Learnable parameter shapes:")
    print(parameter_shape)


def print_simple_stored_facts_block(
    layer_index,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape
):
    print()
    print("Stored Facts:")
    print(f"Layer index: {layer_index}")
    print(f"Layer name: {layer_name}")
    print(f"Layer kind: {layer_kind}")
    print(f"Input shape: {input_shape}")
    print(f"Learnable parameters: {parameter_shape}")
    print(f"Output shape: {output_shape}")
    print()


def print_grouped_stored_facts_block(
    layer_index,
    layer_name,
    layer_kind,
    input_shape,
    output_shape
):
    print()
    print("Stored Facts:")
    print(f"Layer index: {layer_index}")
    print(f"Layer name: {layer_name}")
    print(f"Layer kind: {layer_kind}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    print()


def print_intermediate_tensor_block(title, shape, data_type):
    print()
    print(f"{title}:")
    print(shape)
    print(f"Data type: {data_type}")


def print_simple_layer_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    data_type
):
    print_layer_header(layer_index, layer_title)
    print_shape_block("Input shape", input_shape, data_type)
    print_weight_block(parameter_shape)
    print_shape_block("Output shape", output_shape, data_type)
    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_grouped_layer_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    parameter_shape,
    token_embedding_output_shape,
    positional_embedding_output_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This layer is a grouped layer.")
    print("Internal layers:")
    print(internal_layers)

    print_shape_block_without_type("Input data type", "integer token IDs")
    print_shape_block_without_type("Raw token IDs shape", input_shape)

    print_weight_block(parameter_shape)

    print_intermediate_tensor_block(
        title="Token Embedding output shape",
        shape=token_embedding_output_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Positional Embedding output shape",
        shape=positional_embedding_output_shape,
        data_type="float tensor"
    )

    print()
    print("Combine operation:")
    print(combine_operation)

    print_shape_block("Final Transformer input X shape", output_shape, "float hidden tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_multi_head_attention_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    num_heads,
    head_dim,
    parameter_shape,
    query_shape,
    key_shape,
    value_shape,
    attention_scores_shape,
    attention_weights_shape,
    head_output_shape,
    concatenated_heads_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This layer is a grouped attention layer.")
    print("Internal operations:")
    print(internal_layers)

    print_shape_block("Input hidden tensor shape", input_shape, "float hidden tensor")

    print()
    print("Attention configuration:")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {head_dim}")

    print_weight_block(parameter_shape)

    print_intermediate_tensor_block(
        title="Query tensor Q shape",
        shape=query_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Key tensor K shape",
        shape=key_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Value tensor V shape",
        shape=value_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Attention scores Q × K.T shape",
        shape=attention_scores_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Attention weights softmax output shape",
        shape=attention_weights_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Weighted sum head output shape",
        shape=head_output_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Concatenated heads shape",
        shape=concatenated_heads_shape,
        data_type="float tensor"
    )

    print()
    print("Combine operation:")
    print(combine_operation)

    print_shape_block("Final Multi-Head Attention output shape", output_shape, "float hidden tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_residual_connection_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    source_layer_index,
    parameter_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Residual Connection adds two compatible tensors element-wise.")
    print(f"Residual source layer index: {source_layer_index}")

    print_shape_block("Input tensors", input_shape, "float hidden tensor")

    print_weight_block(parameter_shape)

    print()
    print("Operation:")
    print("previous_layer_output + residual_source_tensor")

    print_shape_block("Residual output shape", output_shape, "float hidden tensor")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_feed_forward_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    activation_name,
    feed_forward_dim,
    parameter_shape,
    first_projection_shape,
    activation_output_shape,
    second_projection_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This layer is a grouped feed-forward layer.")
    print("Internal operations:")
    print(internal_layers)

    print_shape_block("Input hidden tensor shape", input_shape, "float hidden tensor")

    print()
    print("Feed-Forward configuration:")
    print(f"Feed-forward hidden dimension d_ff: {feed_forward_dim}")
    print(f"Selected activation: {activation_name.upper()}")

    print_weight_block(parameter_shape)

    print_intermediate_tensor_block(
        title="First projection output shape",
        shape=first_projection_shape,
        data_type="float hidden tensor"
    )
    print_intermediate_tensor_block(
        title="Activation output shape",
        shape=activation_output_shape,
        data_type="float hidden tensor"
    )
    print_intermediate_tensor_block(
        title="Second projection output shape",
        shape=second_projection_shape,
        data_type="float hidden tensor"
    )

    print()
    print("Combine operation:")
    print(combine_operation)

    print_shape_block("Final Feed-Forward output shape", output_shape, "float hidden tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_activation_layer_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Activation layers apply an element-wise non-linear transformation.")
    print(f"Activation name: {layer_name.upper()}")

    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output tensor shape", output_shape, "float hidden tensor")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_position_wise_linear_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    output_data_type
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Position-wise Linear applies the same trainable projection to every token position.")
    print("Formula: Y = XW + b")

    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output tensor shape", output_shape, output_data_type)

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_softmax_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Softmax converts logits into probabilities over the last dimension.")
    print("Input data type: logits")

    print_shape_block("Input logits shape", input_shape, "logits")
    print_weight_block(parameter_shape)
    print_shape_block("Output probability shape", output_shape, "probabilities")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_hidden_dense_activation_block_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    activation_name,
    hidden_dim,
    parameter_shape,
    projection_shape,
    activation_output_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This layer is a grouped Chapter 6 hidden dense block.")
    print("Internal operations:")
    print(internal_layers)
    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print()
    print("Hidden block configuration:")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Selected activation: {activation_name.upper()}")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Dense projection output shape",
        shape=projection_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Activation output shape",
        shape=activation_output_shape,
        data_type="float hidden tensor",
    )
    print()
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Final hidden block output shape", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def print_feedforward_mlp_block_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    activation_name,
    hidden_dim,
    parameter_shape,
    first_projection_shape,
    activation_output_shape,
    second_projection_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This layer is a generic Chapter 6 feedforward MLP block.")
    print("Internal operations:")
    print(internal_layers)
    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print()
    print("MLP configuration:")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Selected activation: {activation_name.upper()}")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="First projection output shape",
        shape=first_projection_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Activation output shape",
        shape=activation_output_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Second projection output shape",
        shape=second_projection_shape,
        data_type="float hidden tensor",
    )
    print()
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Final MLP output shape", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def print_feedforward_classifier_head_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    classifier_mode,
    hidden_dim,
    num_classes,
    parameter_shape,
    hidden_projection_shape,
    activation_output_shape,
    logits_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This layer is a terminal Chapter 6 feedforward classifier head.")
    print("Internal operations:")
    print(internal_layers)
    print_shape_block("Input feature vector shape", input_shape, "float hidden tensor")
    print()
    print("Classifier configuration:")
    print(f"Classifier mode: {classifier_mode}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of output classes: {num_classes}")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Hidden projection output shape",
        shape=hidden_projection_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Activation output shape",
        shape=activation_output_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Logits output shape",
        shape=logits_shape,
        data_type="logits",
    )
    print()
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Final classifier probability shape", output_shape, "probabilities")
    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def print_embedding_pooling_adapter_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This adapter pools token embeddings into one vector per example.")
    print("Operation: mean over the sequence axis")
    print_shape_block("Input embedding sequence shape", input_shape, "float hidden tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output classifier feature shape", output_shape, "float hidden tensor")
    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape,
    )

from src.utils.printing.embeddings import *
from src.utils.printing.embedding_feature_extraction import *
from src.utils.printing.embedding_similarity import *
from src.utils.printing.embedding_training import *
from src.utils.printing.embedding_analysis import *
from src.utils.printing.classification_heads import *
from src.utils.printing.losses import *
from src.utils.printing.regularization import *
from src.utils.printing.feedforward import *
from src.utils.printing.feedforward_classifier import *
from src.utils.printing.embedding_pooling import *
from src.utils.printing.activations import *
from src.utils.printing.projections import *
from src.utils.printing.transformers import *
from src.utils.printing.speech import *
