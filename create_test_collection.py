import chromadb
from chromadb.config import Settings
import pickle

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/home/iduadmin/Projects/speaker_identifier/chroma"
))

collection = client.create_collection(name="speaker_collection_l2", metadata={"hnsw:space": "l2"})

saved_speaker_embeddings = pickle.load(open("saved_speakers/saved_speaker_embeddings.pkl", "rb"))
saved_speaker_details = pickle.load(open("saved_speakers/saved_speaker_details.pkl", "rb"))
saved_speaker_ids = pickle.load(open("saved_speakers/saved_speaker_ids.pkl", "rb"))


for i in range(0, len(saved_speaker_embeddings)):
    collection.add(ids=[str(collection.count() + 1)], embeddings=saved_speaker_embeddings[i].reshape(1, -1).tolist(), metadatas=[{"speaker_id": saved_speaker_ids[i], "name": saved_speaker_details[saved_speaker_ids[i]]["name"], "details": saved_speaker_details[saved_speaker_ids[i]]["details"]}])
