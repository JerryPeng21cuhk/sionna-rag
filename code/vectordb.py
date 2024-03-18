import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from config import cfg
from functools import partial
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
from pathlib import Path
import json
import typer
from typing_extensions import Annotated


app = typer.Typer(help=__doc__)


class LiteLLMEmbedding(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        response = requests.post(
            f"{cfg['base_url']}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
            json={
                "model": f"{cfg['embedding_model']}",
                "input": input,
            }
        )
        response = response.json()
        return response['data'][0]['embedding']


def read_jsonl(jsonl_fn):
    lines = []
    prev_i = -1
    with open(jsonl_fn, 'r') as reader:
        for line in reader:
            line = json.loads(line)
            if isinstance(line, list):
                assert prev_i < line[0]
                prev_i = line[0]
                line = line[1]
            lines.append(line)
    return lines


class VectorDB:
    def __init__(self, docs_jsonl: Path, embed_jsonl: Path):
        docs = read_jsonl(docs_jsonl)
        embeds = read_jsonl(embed_jsonl)
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(name=cfg['vectordb'], embedding_function=LiteLLMEmbedding())
        collection.add(
            embeddings=embeds,
            documents=docs,
            ids=[f"id{idx}" for idx in range(len(docs))],
        )
        self.collection = collection

    def query(self, text, top_k=5):
        assert isinstance(text, str)
        results = self.collection.query(
            query_texts=[text],
            n_results=top_k,
        )
        return results


@app.command()
def cli(
    docs_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of doc")],
    embed_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of embedding")],
    update_db: Annotated[bool, typer.Option(help="update the vectordb if true")] = False,
):
    assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
    assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
    db = VectorDB(docs_jsonl, embed_jsonl)
    db.query("tensorflow")


if __name__ == "__main__":
    app()
