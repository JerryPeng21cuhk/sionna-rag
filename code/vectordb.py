import chromadb
from config import cfg
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
from pathlib import Path
import json
import typer
from typing_extensions import Annotated
from preprocess import log, LOGGING_HELP
import logging


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


def valid_load(docs_jsonl, embed_jsonl):
    docs = read_jsonl(docs_jsonl)
    embeds = read_jsonl(embed_jsonl)
    assert len(docs) == len(embeds)
    # skip failed embedding
    is_failed = lambda x: isinstance(x[0], str)
    idxs = [i for i,embed in enumerate(embeds) if not is_failed(embed)]
    docs = [docs[idx] for idx in idxs]
    embeds = [embeds[idx] for idx in idxs]
    log.info(f"{len(docs)-len(idxs)} samples are failed and skipped")
    return docs, embeds


class VectorDB:
    def __init__(self, name: str):
        self.name = name  # cfg['vectordb']
        client = chromadb.PersistentClient(path=f"data/{self.name}")
        collection = client.get_or_create_collection(
            name=self.name,
            embedding_function=LiteLLMEmbedding(),
        )
        self.collection = collection
        self.client = client
        log.info(f"VectorDB loaded with {collection.count()} docs")

    def rebuild(self, docs_jsonl: Path, embed_jsonl: Path):
        client = chromadb.PersistentClient(path=f"data/{self.name}")
        if self.name in client.list_collections():
            client.delete_collection(name=self.name)
        collection = client.get_or_create_collection(
            name=self.name,
            embedding_function=LiteLLMEmbedding(),
        )
        docs, embeds = valid_load(docs_jsonl, embed_jsonl)
        collection.upsert(
            embeddings=embeds,
            documents=docs,
            ids=[f"id{idx}" for idx in range(len(docs))],
        )
        self.collection = collection
        self.client = client
        log.info(f"VectorDB reloaded with {collection.count()} docs")

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
    name: Annotated[str, typer.Option(help="database name")] = cfg.get("vectordb", "vectordb"),
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
    assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
    db = VectorDB(name)
    db.rebuild(docs_jsonl, embed_jsonl)
    db.query("tensorflow")


if __name__ == "__main__":
    app()