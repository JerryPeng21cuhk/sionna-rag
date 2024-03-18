from rich import print
from pathlib import Path
import typer
from rich.progress import track
from parallel_request import cli as parallel_cli
from typing_extensions import Annotated
import os
import logging  # for logging rate limit warnings and other messages


def cli(
    input_filepath: Annotated[Path, typer.Argument(help="an jsonl file stores lines of inputs")],
    output_filepath: Annotated[Path, typer.Argument(help="where to store the lines of reponses")],
    model: Annotated[str, typer.Option(help="the LLM/Embedding model to be called")] = "gpt4-1106-preview",
    base_url: Annotated[str, typer.Option(help="URL of the API endpoint to call")] = "https://drchat.xyz",
    api_key: Annotated[str, typer.Option(help="API key to use")] = os.getenv("OPENAI_API_KEY", None),
    max_requests_per_minute: Annotated[int, typer.Option()] = 720 * 0.5,
    max_tokens_per_minute: Annotated[int, typer.Option()] = 300_000 * 0.5,
    token_encoding_name: Annotated[str, typer.Option()] = 'cl100k_base',
    max_attempts: Annotated[int, typer.Option()] = 5,
    logging_level: Annotated[int, typer.Option()] = logging.INFO):
):

