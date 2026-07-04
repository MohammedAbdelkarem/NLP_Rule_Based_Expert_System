from src.utils.printing.common import (
    print_layer_header,
    print_shape_block,
    print_weight_block,
    print_intermediate_tensor_block,
    print_simple_stored_facts_block,
    print_grouped_stored_facts_block,
)


def print_audio_input_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    batch_size,
    duration_seconds,
    sample_rate,
    channels,
    bit_depth,
    sample_count
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Audio Input keeps the channel dimension instead of forcing mono audio.")
    print("The output is a numeric waveform tensor ready for later speech feature extraction layers.")

    print()
    print("Audio configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Duration seconds: {duration_seconds}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Bit depth: {bit_depth}")
    print(f"Samples per channel: {sample_count}")

    print_shape_block("Input audio metadata", input_shape, "audio_file_or_recorded_signal")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Sample count calculation",
        shape=f"duration_seconds × sample_rate = {duration_seconds} × {sample_rate} = {sample_count}",
        data_type="integer_dimension_value"
    )
    print_shape_block("Output waveform tensor shape", output_shape, "float audio waveform tensor")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_framing_windowing_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    sample_rate,
    samples_per_channel,
    frame_length_ms,
    hop_length_ms,
    frame_length_samples,
    hop_length_samples,
    num_frames,
    channels,
    window_function
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Framing + Windowing converts a continuous waveform into short overlapping frames.")
    print("The channel dimension is preserved instead of collapsing the audio to mono.")

    print()
    print("Framing configuration:")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples per channel: {samples_per_channel}")
    print(f"Frame length: {frame_length_ms} ms = {frame_length_samples} samples")
    print(f"Hop length: {hop_length_ms} ms = {hop_length_samples} samples")
    print(f"Window function: {window_function}")
    print(f"Number of frames: {num_frames}")
    print(f"Channels preserved: {channels}")

    print_shape_block("Input waveform tensor shape", input_shape, "float audio waveform tensor")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Frame length calculation",
        shape=f"sample_rate × frame_length_ms / 1000 = {sample_rate} × {frame_length_ms} / 1000 = {frame_length_samples}",
        data_type="integer_dimension_value"
    )
    print_intermediate_tensor_block(
        title="Hop length calculation",
        shape=f"sample_rate × hop_length_ms / 1000 = {sample_rate} × {hop_length_ms} / 1000 = {hop_length_samples}",
        data_type="integer_dimension_value"
    )
    print_intermediate_tensor_block(
        title="Number of frames calculation",
        shape=f"1 + floor(({samples_per_channel} - {frame_length_samples}) / {hop_length_samples}) = {num_frames}",
        data_type="integer_dimension_value"
    )
    print_intermediate_tensor_block(
        title="Window vector shape",
        shape=f"({frame_length_samples},)",
        data_type=f"float {window_function} window vector"
    )
    print_shape_block("Output framed audio tensor shape", output_shape, "float framed audio tensor")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )



def print_stft_spectrum_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    sample_rate,
    n_fft,
    output_mode,
    num_frames,
    frequency_bins,
    channels
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a speech feature-extraction component group.")
    print("STFT / Spectrum Extraction converts each windowed frame from time samples into frequency bins.")
    print("It preserves both the frame dimension and the channel dimension.")
    print("It uses a fixed Fourier transform, so it has no trainable weights.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("STFT configuration:")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"n_fft: {n_fft}")
    print(f"Output mode: {output_mode}")
    print(f"Frequency bins: floor({n_fft} / 2) + 1 = {frequency_bins}")
    print(f"Frames preserved: {num_frames}")
    print(f"Channels preserved: {channels}")

    print_shape_block("Input framed audio tensor shape", input_shape, "float framed audio tensor")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Frequency-bin calculation",
        shape=f"floor(n_fft / 2) + 1 = floor({n_fft} / 2) + 1 = {frequency_bins}",
        data_type="integer_dimension_value"
    )
    print_intermediate_tensor_block(
        title="Fixed Fourier basis shape",
        shape=f"({frequency_bins}, {n_fft})",
        data_type="fixed complex Fourier basis, not trainable"
    )
    print_shape_block("Output spectrum tensor shape", output_shape, f"{output_mode} frequency spectrum tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_mel_filterbank_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    sample_rate,
    nyquist_frequency_hz,
    num_mel_filters,
    min_frequency_hz,
    max_frequency_hz,
    num_frames,
    frequency_bins,
    channels
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a speech feature-extraction component group.")
    print("Mel Filterbank converts a linear-frequency spectrum into Mel frequency bands.")
    print("It preserves both the frame dimension and the channel dimension.")
    print("It uses a fixed triangular Mel filterbank matrix, so it has no trainable weights.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("Mel Filterbank configuration:")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Nyquist frequency: {nyquist_frequency_hz} Hz")
    print(f"Frequency range: {min_frequency_hz} Hz → {max_frequency_hz} Hz")
    print(f"Input frequency bins: {frequency_bins}")
    print(f"Number of Mel filters: {num_mel_filters}")
    print(f"Frames preserved: {num_frames}")
    print(f"Channels preserved: {channels}")

    print_shape_block("Input spectrum tensor shape", input_shape, "magnitude/power frequency spectrum tensor")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Fixed Mel filterbank matrix shape",
        shape=f"({frequency_bins}, {num_mel_filters})",
        data_type="fixed float triangular Mel filterbank matrix, not trainable"
    )
    print_intermediate_tensor_block(
        title="Matrix projection",
        shape=(
            f"(B, frames, {frequency_bins}, channels) × "
            f"({frequency_bins}, {num_mel_filters}) → "
            f"(B, frames, {num_mel_filters}, channels)"
        ),
        data_type="fixed_linear_frequency_to_mel_projection"
    )
    print_shape_block("Output Mel spectrum tensor shape", output_shape, "mel frequency spectrum tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_log_mel_spectrogram_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    epsilon,
    compression_mode,
    sample_rate,
    num_frames,
    num_mel_filters,
    channels
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Log-Mel Spectrogram applies element-wise log compression to Mel energy features.")
    print("It preserves the frame dimension, Mel-filter dimension, and channel dimension.")
    print("It has no trainable weights, no bias vector, and no matrix multiplication.")

    print()
    print("Log-Mel configuration:")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Compression mode: {compression_mode}")
    print(f"Epsilon: {epsilon}")
    print(f"Frames preserved: {num_frames}")
    print(f"Mel filters preserved: {num_mel_filters}")
    print(f"Channels preserved: {channels}")

    if compression_mode == "decibel":
        formula = "10 * log10(mel_energy + epsilon)"
    elif compression_mode == "log10":
        formula = "log10(mel_energy + epsilon)"
    else:
        formula = "log(mel_energy + epsilon)"

    print_shape_block("Input Mel spectrum tensor shape", input_shape, "mel frequency spectrum tensor")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Element-wise log compression",
        shape=f"{formula}, epsilon={epsilon}",
        data_type="fixed_elementwise_log_function_not_trainable"
    )
    print_intermediate_tensor_block(
        title="Shape preservation",
        shape=f"{input_shape} → {output_shape}",
        data_type="same_rank_same_dimensions"
    )
    print_shape_block("Output Log-Mel spectrogram tensor shape", output_shape, "log-mel spectrogram tensor")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )



def print_mfcc_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    sample_rate,
    num_frames,
    num_mel_filters,
    num_mfcc_coefficients,
    dct_type,
    include_c0,
    coefficient_range,
    lifter,
    channels
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a speech feature-extraction component group.")
    print("MFCC converts a Log-Mel spectrogram into cepstral coefficients using a fixed DCT matrix.")
    print("It reduces the feature dimension while preserving batch, frame, and channel dimensions.")
    print("It has a fixed matrix, but the matrix is not trainable.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("MFCC configuration:")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Input Mel filters: {num_mel_filters}")
    print(f"MFCC coefficients: {num_mfcc_coefficients}")
    print(f"DCT type: {dct_type}")
    print(f"C0 policy: {include_c0}")
    print(f"Coefficient range: {coefficient_range}")
    print(f"Cepstral lifter: {lifter}")
    print(f"Frames preserved: {num_frames}")
    print(f"Channels preserved: {channels}")

    print_shape_block("Input Log-Mel spectrogram tensor shape", input_shape, "log-mel spectrogram tensor")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Fixed DCT matrix shape",
        shape=f"({num_mel_filters}, {num_mfcc_coefficients})",
        data_type="fixed float DCT matrix, not trainable"
    )
    print_intermediate_tensor_block(
        title="Matrix projection",
        shape=(
            f"(B, frames, {num_mel_filters}, channels) × "
            f"({num_mel_filters}, {num_mfcc_coefficients}) → "
            f"(B, frames, {num_mfcc_coefficients}, channels)"
        ),
        data_type="fixed_dct_projection_over_mel_axis"
    )
    print_shape_block("Output MFCC tensor shape", output_shape, "mfcc feature tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_delta_features_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    sample_rate,
    num_frames,
    num_mfcc_coefficients,
    concatenated_coefficients,
    delta_window,
    adapter_shape,
    channels
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a speech feature-extraction component group.")
    print("It always outputs three separated streams: MFCC, Delta, and Delta-Delta.")
    print("The temporal kernels are fixed, so they are not trainable weights.")
    print("A built-in adapter is available, but it is only suggested when the next layer needs one concatenated tensor.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("Delta feature configuration:")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Frames preserved: {num_frames}")
    print(f"MFCC coefficients per stream: {num_mfcc_coefficients}")
    print(f"Concatenated adapter coefficients: {concatenated_coefficients}")
    print(f"Delta window radius: {delta_window}")
    print("Primary output design: separate_streams_always")
    print(f"Channels preserved: {channels}")

    print_shape_block("Input MFCC tensor shape", input_shape, "mfcc feature tensor")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Fixed delta temporal kernel",
        shape=f"uses neighboring frames from t-{delta_window} to t+{delta_window}",
        data_type="fixed temporal-difference kernel, not trainable"
    )
    print_intermediate_tensor_block(
        title="Separate primary output streams",
        shape=(
            f"MFCC = (B, frames, {num_mfcc_coefficients}, channels), "
            f"Delta = (B, frames, {num_mfcc_coefficients}, channels), "
            f"Delta-Delta = (B, frames, {num_mfcc_coefficients}, channels)"
        ),
        data_type="separate static / velocity / acceleration feature streams"
    )
    print_intermediate_tensor_block(
        title="Available concatenation adapter",
        shape=adapter_shape,
        data_type="adapter output tensor, not trainable; suggest only when the next layer requires one tensor"
    )
    print_shape_block("Primary output shape", output_shape, "separate_delta_feature_stream_group")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_feature_normalization_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    normalization_mode_choice,
    normalization_mode_name,
    num_frames,
    num_features,
    concatenated_features,
    channels,
    statistics_shape,
    adapter_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a speech feature-normalization layer.")
    print("It normalizes the separated MFCC, Delta, and Delta-Delta streams.")
    print("The primary output remains separated; concatenation is only an adapter suggestion for later layers.")
    print("It stores dataset statistics, but has no trainable weights, bias, or matrix.")

    print()
    print("Normalization configuration:")
    print(f"Mode choice: {normalization_mode_choice}")
    print(f"Mode: {normalization_mode_name}")
    print(f"Frames preserved: {num_frames}")
    print(f"Feature coefficients per stream: {num_features}")
    print(f"Concatenated adapter coefficients: {concatenated_features}")
    print(f"Channels preserved: {channels}")
    print(f"Statistics shape per stream: {statistics_shape}")

    if normalization_mode_name == "range_scaling_minus_one_to_one":
        operation = "x_norm = 2 * (x - min) / (max - min + epsilon) - 1"
        statistics = "min/max statistics"
    else:
        operation = "x_norm = (x - mean) / (std + epsilon)"
        statistics = "mean/std statistics"

    print_shape_block("Input separated Delta feature stream group", input_shape, "separate delta feature stream group")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Normalization operation",
        shape=operation,
        data_type="fixed element-wise feature normalization, not trainable"
    )
    print_intermediate_tensor_block(
        title="Stored statistics",
        shape=f"{statistics} for each stream with shape {statistics_shape}",
        data_type="stored training-dataset statistics, not trainable weights"
    )
    print_intermediate_tensor_block(
        title="Separated normalized streams",
        shape=(
            f"Normalized MFCC = (B, frames, {num_features}, channels), "
            f"Normalized Delta = (B, frames, {num_features}, channels), "
            f"Normalized Delta-Delta = (B, frames, {num_features}, channels)"
        ),
        data_type="normalized separate static / velocity / acceleration streams"
    )
    print_intermediate_tensor_block(
        title="Available concatenation adapter",
        shape=adapter_shape,
        data_type="adapter suggestion only; use when the next layer requires one tensor"
    )
    print_shape_block("Primary output shape", output_shape, "normalized separate delta feature stream group")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )
