from importlib.metadata import version
import tiktoken

print("tiktoken version: ", version("tiktoken"))

# BPE (Byte Pair Encoding) tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)
