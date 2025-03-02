from .read_verdict import read

raw_text = read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])
