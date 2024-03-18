import typer
from typing_extensions import Annotated
from rich import print
from rich.progress import track
from tiktoken import get_encoding
from collections import OrderedDict


def to_json(filepath: str):
    with open(filepath, 'r') as reader:
        lines = reader.read()
        dictified = markdown_to_json.dictify(lines)
        return dictified


def get_token(input, encoding="cl100k_base"):
    encoder = get_encoding(encoding)
    return len(encoder.encode(input))


def chunk(json, max_size=1000, min_size=50):
    def traverse(json, indent=""):
        ck_size = 0
        if isinstance(json, str):
            # if min_size <= get_token(json) <= max_size:
            ck_size += get_token(json)
            writer.write('\n'+"="*80+'\n')
            writer.write(indent+json)
            writer.write('\n'+"="*80+'\n')
        elif isinstance(json, list):
            for item in json:
                ck_size += traverse(item, indent+"  ")
        elif isinstance(json, OrderedDict):
            for key, subjson in json.items():
                ck_size += traverse(subjson, indent)
        else:
            raise NotImplementedError(f"Cannot handle {type(json)}")
        return ck_size
    with open("data/trash/debug.md", 'w') as writer:
        sz = traverse(json)
    return sz


if __name__ == "__main__":
    # json = to_json("data/clean/tutorials_for_beginners_basic_MIMO_simulations.md")
    json = to_json("data/trash/input.md")
    print(chunk(json))