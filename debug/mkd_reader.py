import re
from functools import partial
import typer
from typing_extensions import Annotated
from rich import print
from rich.progress import track


app = typer.Typer(help="""
This script cleans markdown files by removing noisy texts, image urls, etc.
""")


def remove_leading_content(lines):
    seen_title = False
    for line in lines:
        if seen_title:
            yield line
            continue
        if line[0] == '#':
            seen_title = True
            yield line


def remove_headerlink(lines):
    pattern = re.compile('<a[^>]*class="headerlink"[^>]*>.*?</a>')
    for line in lines:
        # Removing the matched pattern from the string
        cleaned_line = pattern.sub('', line)
        if cleaned_line == '': continue
        yield cleaned_line


def remove_image(lines):
    pattern = re.compile('^<img ')
    for line in lines:
        if pattern.match(line): continue
        yield line


def remove_toc(lines):
    is_start = False
    start_pattern = re.compile('## Table of Contents')
    end_pattern = re.compile('## ')
    for line in lines:
        if not is_start and start_pattern.match(line):
            is_start = True
        elif is_start and end_pattern.match(line):
            is_start = False
        if not is_start: yield line


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


remove_codeblock_idx = partial(multiline_match, patterns=[
    "^```python$",
    "^\[\d+\]:$",
    "^```$",
])


class MarkdownReader(object):
    def __init__(self, filepath: str):
        pipeline = [
            remove_leading_content,
            remove_image,
            remove_toc,
            remove_headerlink,
            remove_codeblock_idx,
        ]
        self.stream = open(filepath)
        iterator = self.stream
        for func in pipeline:
            iterator = func(iterator)
        self.iterator = iterator

    def __enter__(self):
        return self.iterator

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.close()
        

@app.command("single")
def single(
    input_file: Annotated[str, typer.Argument(help="the input markdown file")] = \
        "data/markdown/tutorials_for_beginners_part1.md",
    output_file: Annotated[str, typer.Argument(help="the output markdown file")] = \
        "data/trash/debug.md",
):
    """
    Clean a single markdown file.
    """
    with open(output_file, 'w') as writer, \
        MarkdownReader(input_file) as reader:
        for line in reader:
            writer.write(line)


@app.command("batch")
def batch(
    input_dir: Annotated[str, typer.Argument(help="the input dir that contains markdown files")] = \
        "data/markdown",
    output_dir: Annotated[str, typer.Argument(help="the output dir that stores processed markdown file")] = \
        "data/clean"
):
    """
    Clean all markdown files inside a dir
    """
    import glob
    import os
    files = glob.glob(f"{input_dir}/**/*.md", recursive=True)
    for input_file in track(
            files,
            description=f"Formatting {len(files)} markdown files from {input_dir}"):
        base = input_file.removeprefix(input_dir)
        output_file = f"{output_dir}/{base}"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        single(input_file, output_file)
    print(f"Saved to {output_dir}")
    

if __name__ == "__main__":
   app()