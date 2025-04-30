import torch
from matplotlib import pyplot as plt
import numpy as np


def minmax_normalize(matrix, per_row=False):
    if per_row:
        # Min-Max normalization per row to [0, 1]
        row_min = matrix.min(axis=1, keepdims=True)
        row_max = matrix.max(axis=1, keepdims=True)
        denom = row_max - row_min
        # Avoid division by zero
        denom[denom == 0] = 1
        normalized_matrix = (matrix - row_min) / denom
        return normalized_matrix
    else:
        # Min-Max normalization over the entire matrix to [0, 1]
        matrix_min = matrix.min()
        matrix_max = matrix.max()
        denom = matrix_max - matrix_min
        if denom > 0:
            return (matrix - matrix_min) / denom
        else:
            return np.zeros_like(matrix)
        
def normalize(matrix, norm_type="minmax", clip_value=1e6):
    """
    Normalize the attention matrix.

    Parameters:
    - matrix (np.ndarray): The attention matrix to normalize.
    - norm_type (str): The type of normalization to apply ('minmax', 'tanh').
    - clip_value (float): The value to clip the matrix elements to avoid overflow.
    - per_row (bool): If True, apply normalization per row. If False, apply over the entire matrix.

    Returns:
    - np.ndarray: The normalized attention matrix.
    """

    if np.isnan(matrix).any():
        print("NaNs detected in the attention matrix. Replacing with 0.")
        # Replace NaNs with 0 for normalization purposes
        matrix = np.nan_to_num(matrix, nan=0.0)
    
    # Clip extreme values to avoid overflow
    matrix = np.clip(matrix, -clip_value, clip_value)

    if norm_type == "minmax_per_row":
        return minmax_normalize(matrix, per_row=True)
    
    elif norm_type == "minmax_overall":
        return minmax_normalize(matrix, per_row=False)

    elif norm_type == "tanh":
        # Tanh normalization to [-1, 1], keeping 0 mapped to 0
        return np.tanh(matrix)

    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

def ensure_axis_2d(axs, num_layers):
    """
    Ensure that the input axis is a 2D array.
    """
    if isinstance(axs, np.ndarray):
        if axs.ndim == 1:
            if num_layers == 1:
                axs = axs.reshape(1, -1)
            else:
                axs = axs.reshape(-1, 1)
    else:
        # axs is a single AxesSubplot object
        axs = np.array([[axs]])
    return axs


def plot_attention(model, tokenizer, gather, aggregate, input_text):

    # Tokenize the input text
    norm_type = "minmax_overall"  # Normalization type: 'minmax_per_row', 'minmax_overall', 'tanh'
    tokenizer.padding_side = "right"
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True).to('cuda')['input_ids']
    tokens = [tokenizer.decode(i) for i in input_ids[0]]
    tokens = [t if t != '\n' else '\\n' for t in tokens]

    for layer_idx, head_idx in [gather, aggregate]:
        model.backbone.layers[layer_idx].mixer.materialize_heads = [head_idx]
        model.backbone.layers[layer_idx].mixer.chunk_size = len(input_ids[0])
    
    # Forward pass through the model
    choices = ['A', 'B', 'C', 'D']
    choices_tokens = {f"{ch}": tokenizer.encode(f"{ch}", add_special_tokens=False)[0] for ch in choices}
    attn_implementation = model.config._attn_implementation
    model.config._attn_implementation = 'eager'
    with torch.inference_mode():
        output = model(input_ids, output_attentions=True)
        logits = output.logits.squeeze()
        lprobs = torch.tensor([logits[-1, choices_tokens[ch]] for ch in choices]).to(torch.float32)
        pred = choices[int(torch.argmax(lprobs))]
    model.config._attn_implementation = attn_implementation

    gather_matrices = {(gather[0], head_idx): output.attentions[gather[0]][0,head_idx] for head_idx in gather[1]}
    aggregate_matrices = {(aggregate[0], head_idx): output.attentions[aggregate[0]][0,head_idx] for head_idx in aggregate[1]}
    matrices = {**gather_matrices, **aggregate_matrices}
    num_heads = matrices[gather[0], gather[1][0]].shape[1]
    
    for k in matrices:
        matrices[k] = matrices[k].cpu().detach().to(torch.float32).numpy()
        matrices[k] = normalize(matrices[k], norm_type=norm_type)

    max_heads_per_layer = max(len(gather[1]), len(aggregate[1]))
    
    size_factor = 3  # Increase the size of the plot
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=max_heads_per_layer, 
        # figsize=(max_heads_per_layer * size_factor, 2 * size_factor),
        figsize=(max_heads_per_layer * size_factor, 2 * size_factor),
        dpi=250,
        )
    axs = ensure_axis_2d(axs, 2)

    plt.tight_layout(h_pad=5, w_pad=2, rect=[0, 0, 1, 0.85])  # Leave space for the title

    # Iterate through each layer
    for (layer_idx, head_idx), matrix in matrices.items():

        row_idx = [key[0] for key in matrices.keys()].index(layer_idx)
        col_idx = sorted([x[1] for x in matrices.keys() if x[0] == layer_idx]).index(head_idx)
        ax = axs[row_idx][col_idx]

        im = ax.imshow(matrix, cmap="viridis", interpolation="nearest", vmin=matrix.min(), vmax=matrix.max())
        ax.set_title(f"Layer {layer_idx}/{len(model.backbone.layers) -1}\n"
                     f"Head {head_idx}/{num_heads}")

        # Set tokens as labels on x and y axes
        seq_len = matrix.shape[0]
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, fontsize=3)  # x-axis labels
        ax.set_yticklabels(tokens, fontsize=3)  # y-axis labels

    # Turn off any subplot that has no data
    for ax in axs.ravel():
        if not ax.has_data():
            ax.axis('off')

    # Add a colorbar to one of the subplots
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)

    # Show the entire grid as one image
    plt.show()
    plt.close()
