import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from config import cfg
from functools import partial
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests


class LiteLLMEmbedding(EmbeddingFunction):
    async def __call__(self, input: Documents) -> Embeddings:
        response = requests.post(
            f"{cfg['base_url']}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
            json={
                "model": "text-embedding-3-small",
                "input": input,
            }
        )
        response = response.json()
        return response['data'][0]['embedding']


# client = chromadb.PersistentClient(path="/path/to/save/to")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection", embedding_function=LiteLLMEmbedding())
collection.add(
    documents=["This is A document"],
    metadatas=[{"source": "my_source"}],
    ids=["id1"]
)
collection.add(
    documents=["This is not a document"],
    metadatas=[{"source": "my_source"}],
    ids=["id2"]
)

results = collection.query(
    query_texts=["This is a document"],
    n_results=2
)