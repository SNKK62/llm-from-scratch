with open("the-verdict.txt", "r") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])
