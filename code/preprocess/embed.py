from clean import MarkdownReader
from chunk import MdTree
from rich import print
from pathlib import Path
import typer


app = typer.Typer(help=__doc__)


def process_one_file(input: Path, output: Path):
    assert input != output, f"Input and output path ({input}) cannot be the same."
    with open(output, 'a') as writer, \
        MarkdownReader(input) as reader:
        for line in MdTree(reader):
            