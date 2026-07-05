from src.facts import CalculationStep, Layer, LayerMetadata, TensorInfo, WeightShape


def vector_shape(vector_dim):
    return f"({int(vector_dim)},)"


def declare_similarity_component(
    engine,
    layer_index,
    layer_title,
    vector_dim,
    input_shape,
    output_shape,
    parameter_shape,
):
    engine.declare(Layer(
        layer_index=layer_index,
        layer_name="cosine_similarity",
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="cosine_similarity",
        tensor_name="cosine_similarity_score",
        role="output",
        dims_count=1,
        shape=output_shape,
        data_type="float_similarity_score",
        source="cosine_similarity"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="cosine_similarity",
        tensor_name="vector_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(vector_dim)),
        data_type="integer_dimension_value",
        source="cosine_similarity"
    ))
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name="cosine_similarity",
        weight_name="learnable_parameters",
        shape=parameter_shape
    ))
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name="cosine_similarity",
        layer_kind="embedding_similarity_component_not_neural_layer",
        layer_role="compare_two_embedding_vectors",
        repeat_policy="allowed",
        consecutive_repeat_policy="allowed_with_new_vector_pair",
        terminal_status="terminal"
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="cosine_similarity",
        step_key="cosine_similarity_scalar_shape",
        previous_step_key="embedding_vector_pair",
        step_order=1,
        title="Normalize dot product between two embedding vectors",
        formula="cosine_similarity_output_shape = (1,)",
        substitution=f"vector_dim = {int(vector_dim)}",
        result=f"cosine_similarity_output_shape = {output_shape}"
    ))
