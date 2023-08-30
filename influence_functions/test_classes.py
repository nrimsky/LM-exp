import torch as t
from decoder_only_transformer import MultiQueryAttention, DecoderTransformer


def test_multi_query_attention():
    # Create a dummy input tensor X with batch size=32, sequence length=10, and d_model=64
    X = t.rand((32, 10, 64))

    # Initialize MultiQueryAttention
    n_heads = 4  # You can choose the number of heads
    multi_query_attn = MultiQueryAttention(d_model=64, n_heads=n_heads)

    # Forward pass
    output = multi_query_attn(X)

    # Print the shape of the output
    print(output.shape)  # Expected shape: [32, 10, 64]

    assert output.shape == (32, 10, 64)

def test_transformer():
    # Hyperparameters
    d_model = 512
    n_heads = 8
    d_mlp = 2048
    n_layers = 6
    vocab_size = 1000
    batch_size = 32
    sequence_length = 20

    # Initialize the model
    model = DecoderTransformer(d_model, n_heads, d_mlp, n_layers, vocab_size)

    # Move the model to 'cuda' if available
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create a dummy input tensor
    dummy_input = t.randint(0, vocab_size, (batch_size, sequence_length)).to(device)

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Check the output shape and values
    print(f"Output Shape: {output.shape}")
    assert output.shape == (batch_size, sequence_length, vocab_size)
    print(f"Sample Output Values:\n{output[0, 0, :5]}")  # First 5 values for the first sequence's first token

    # Ensure there are no NaN or Inf values in the output
    assert not t.isnan(output).any().item()
    assert not t.isinf(output).any().item()
    print("Output has valid values!")



if __name__ == "__main__":
    test_multi_query_attention()
    test_transformer()
