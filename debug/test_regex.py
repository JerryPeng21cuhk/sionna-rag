import re
from functools import partial


def multiline_match(stream, patterns):
    """
    Process a large text stream, removing patterns that span multiple lines.

    :param stream: An iterable text stream (e.g., a file object).
    :param pattern: A compiled regex pattern designed to match across multiple lines.
    """
    patterns = [re.compile(p) for p in patterns]
    buffer = []
    for line in stream:
        if patterns[len(buffer)].match(line):
            buffer.append(line)
            if len(buffer) == len(patterns):
                buffer = []
            continue
        for l in buffer:
            yield l
        buffer = []
        if patterns[0].match(line):
            buffer = [line]
        else:
            yield line
    for line in buffer:
        yield line


remove_code_idx = partial(multiline_match, patterns=[
    "^```python$",
    "^\[\d+\]:$",
    "^```$",
])
# Example usage with a file (or any line-based text stream)
# file_path = "your_file_path_here.txt"

input_lines = [
    "Some text before the pattern.",
    "```python",
    "[2432]:",
    "```python",
    "[2432]:",
    "```",
    "```python",
    "[2432]:",
    "```python",
    "```python",
    "[]:",
    "```",
    "Some text after the pattern.",
    "```python",
    "[2432]:",
]

# with open(file_path, 'r') as file:
for cleaned_chunk in remove_code_idx(iter(input_lines)):
    print(cleaned_chunk)
