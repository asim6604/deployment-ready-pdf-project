import chromadb
import uuid
from parsing import chunking
from embedding import embedding
import hashlib

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def Vector(embeddings, chunks, FileObject):
    file_hash = hashlib.sha256(FileObject).hexdigest()
    collection = chroma_client.get_or_create_collection(name=file_hash)
    List_embed = embeddings.tolist()
    length = len(chunks)
    id = []
    for i in range(0, length, 1):
        n = str(i)
        id.append(n)
    collection.add(
        ids=id,
        documents=chunks,
        embeddings=List_embed
    )
    return file_hash
def Get_Answer(embeddings,FileObject):
     embeddings=embeddings.tolist();
     collection = chroma_client.get_collection(name=FileObject)
     results = collection.query(
        query_embeddings=embeddings,
        n_results=5
)
     return results
