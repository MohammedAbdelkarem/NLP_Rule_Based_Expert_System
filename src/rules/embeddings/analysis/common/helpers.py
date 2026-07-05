from src.facts import CalculationStep, ComponentGroup, LayerMetadata, TensorInfo, WeightShape


def matrix_shape(row_count, column_count):
    return f"({int(row_count)}, {int(column_count)})"


def declare_analysis_component_group(
    engine,
    layer_index,
    group_name,
    group_title,
    layer_kind,
    layer_role,
    internal_layers,
    combine_operation,
    input_shape,
    output_shape,
    tensor_name,
    tensor_type,
    parameter_name,
    parameter_shape,
):
    engine.declare(ComponentGroup(
        group_index=layer_index,
        group_name=group_name,
        group_title=group_title,
        internal_layers=internal_layers,
        combine_operation=combine_operation,
        input_shape=input_shape,
        output_shape=output_shape,
        status="computed"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name=tensor_name,
        role="output",
        dims_count=2,
        shape=output_shape,
        data_type=tensor_type,
        source=group_name
    ))
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=group_name,
        weight_name=parameter_name,
        shape=parameter_shape
    ))
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=group_name,
        layer_kind=layer_kind,
        layer_role=layer_role,
        repeat_policy="allowed",
        consecutive_repeat_policy="allowed_with_new_embedding_set",
        terminal_status="terminal"
    ))


def declare_visualization_metadata(
    engine,
    layer_index,
    num_items,
    embedding_dim,
    visualization_dim,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_visualization",
        tensor_name="num_items",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(num_items)),
        data_type="integer_dimension_value",
        source="embedding_visualization"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_visualization",
        tensor_name="embedding_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(embedding_dim)),
        data_type="integer_dimension_value",
        source="embedding_visualization"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_visualization",
        tensor_name="visualization_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(visualization_dim)),
        data_type="integer_dimension_value",
        source="embedding_visualization"
    ))


def declare_bias_metadata(
    engine,
    layer_index,
    target_word_count,
    attribute_word_count,
    embedding_dim,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_bias_analysis",
        tensor_name="target_word_count",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(target_word_count)),
        data_type="integer_dimension_value",
        source="embedding_bias_analysis"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_bias_analysis",
        tensor_name="attribute_word_count",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(attribute_word_count)),
        data_type="integer_dimension_value",
        source="embedding_bias_analysis"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_bias_analysis",
        tensor_name="embedding_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(embedding_dim)),
        data_type="integer_dimension_value",
        source="embedding_bias_analysis"
    ))


def declare_evaluation_metadata(
    engine,
    layer_index,
    evaluation_pair_count,
    embedding_dim,
    benchmark_count,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_evaluation",
        tensor_name="evaluation_pair_count",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(evaluation_pair_count)),
        data_type="integer_dimension_value",
        source="embedding_evaluation"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_evaluation",
        tensor_name="embedding_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(embedding_dim)),
        data_type="integer_dimension_value",
        source="embedding_evaluation"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="embedding_evaluation",
        tensor_name="benchmark_count",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(benchmark_count)),
        data_type="integer_dimension_value",
        source="embedding_evaluation"
    ))


def declare_analysis_step(engine, layer_index, layer_name, step_title, output_shape):
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"{layer_name}_output_shape",
        previous_step_key="embedding_matrix_input",
        step_order=1,
        title=step_title,
        formula="analysis_output_shape = declared_report_shape",
        substitution=f"analysis_output_shape = {output_shape}",
        result=f"analysis_output_shape = {output_shape}"
    ))
