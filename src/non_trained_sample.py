from utils import generate_text_simple, text_to_token_ids, token_ids_to_text


if __name__ == "__main__":
    import tiktoken
    from models import GPTModel

    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Hello, I am"
    print("start_context:", start_context)
    encoded_tensor = text_to_token_ids(start_context, tokenizer)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    model = GPTModel(GPT_CONFIG_124M)

    model.eval()
    out = generate_text_simple(
        model, encoded_tensor, 6, GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = token_ids_to_text(out, tokenizer)
    print("Decoded text:", decoded_text)
