import os


def read():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, "the-verdict.txt")
    with open(file_path, "r") as f:
        raw_text = f.read()
    return raw_text
