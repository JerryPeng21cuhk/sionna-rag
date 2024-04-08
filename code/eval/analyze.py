from config import cfg
from pathlib import Path
import re
import json
import typer
from typing_extensions import Annotated
from preprocess import log, LOGGING_HELP
import logging
from prompt import score_prompt


app = typer.Typer(help=__doc__)


def read_jsonl(jsonl_fn):
    with open(jsonl_fn, 'r') as reader:
        for line in reader:
            json_data = json.loads(line)
            yield json_data


def parse_scores(lines):
    pattern = re.compile(r'^Correctness Score: (\d+)')
    scores = []
    for line in lines:
        score = pattern.match(line)[1]
        scores.append(int(score))
    return scores


@app.command("compare")
def compare(
    score1_jsonl: Annotated[Path, typer.Argument(help="path to a evaluation.jsonl file")],
    score2_jsonl: Annotated[Path, typer.Argument(help="path to another evaluation.jsonl file")],
    figure: Annotated[Path, typer.Argument(help="path to save the figure")],
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    scores1 = parse_scores(read_jsonl(score1_jsonl))
    scores2 = parse_scores(read_jsonl(score2_jsonl))
    # for s1, s2 in zip(score1_iter, score2_iter):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.scatter(scores1, scores2, alpha=0.5)
    plt.savefig(figure)



if __name__ == "__main__":
    app()
