# Libraries
import traceback
import chromadb
from chromadb.config import Settings
from fastapi import UploadFile, Path, Query
from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

# Scripts
from split_speakers import SplitSpeakers
from identify_speaker import IdentifySpeaker

# Setup API
app = FastAPIOffline(
    title="Speaker identifier"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Adjust below to change how often speakers are saved to disk
PERSIST_ENCODING_MULTIPLE = 5
count = 0

# How many speakers to get with collection query
N_SPEAKERS = 3

# Load saved speakers
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chroma",
))
collection = client.get_collection(name="speaker_collection_l2")

# Models
diarise_model = SplitSpeakers()


# Identify speakers and persist if PERSIST_ENCODING_MULTIPLE is reached
def identify_speaker(file_path, model, n_speakers=N_SPEAKERS):
    global count, collection, client

    embedding = model.create_embedding(file_path)
    if embedding == "RuntimeError":
        return embedding

    identified_speakers = model.identify_speaker(embedding, collection, n_speakers)
    identified_speakers.append(("saved_id " + str(collection.count())))

    count += 1

    # Save speakers to disk
    if count % PERSIST_ENCODING_MULTIPLE == 0:
        persist()
        print("saved to disk")

    return identified_speakers


# Upload audio with multiple speakers
@app.post("/upload_audio")
async def upload_audio_file(audio_file: UploadFile, save_and_identify: bool = False):
    global diarise_model, count, collection

    tmp_path = save_upload_file_tmp(audio_file)

    # Split and save audio
    count, result = diarise_model.split_and_save_audio(tmp_path, count)

    # Save and identify speakers in audio files
    if save_and_identify is True:
        identify_model = IdentifySpeaker()

        list_of_speakers = {}
        distances = {}
        all_results = {}

        # For each individual speaker
        for directory in os.listdir("audio_files"):
            list_of_speakers[directory] = {}
            distances[directory] = {}
            all_results[directory] = {}

            # For each audio file
            for file in os.listdir(os.path.join("audio_files", directory)):
                file_path = os.path.join("audio_files", directory, file)
                try:
                    # identify_speakers(audio_file_path, model, top_n_speakers_identified)
                    identified_speakers = identify_speaker(file_path, identify_model, 1)

                except:
                    print("file:", file)
                    traceback.print_exc()
                    continue

                # If audio file is too short
                if identified_speakers == "RuntimeError":
                    continue

                for speaker in identified_speakers:
                    try:
                        list_of_speakers[directory][str(speaker["speaker_id"])] += 1
                        distances[directory][str(speaker["speaker_id"])] += speaker["distance"]
                    # If new speaker
                    except KeyError:
                        list_of_speakers[directory][str(speaker["speaker_id"])] = 1
                        distances[directory][str(speaker["speaker_id"])] = speaker["distance"]

                    # If reached the end of identified speakers
                    except TypeError:
                        pass

            # Output the most common identified speaker per directory
            all_results[directory]["speaker_id"] = max(list_of_speakers[directory], key=list_of_speakers[directory].get)
            all_results[directory]["speaker_name"] = collection.get(where={"speaker_id": all_results[directory]["speaker_id"]})["metadatas"][0]["name"]

            # Calculate average distance of identified speaker
            all_results[directory]["distance"] = distances[directory][all_results[directory]["speaker_id"]] / max(list_of_speakers[directory].values())

        return all_results

    return result


# Identify a speaker in an audio file with a single speaker in it
@app.post("/identify_speaker", description="Identify single speaker in audio file")
async def identify_single_speaker(audio_file: UploadFile):
    global count, collection, N_SPEAKERS

    tmp_path = save_upload_file_tmp(audio_file)

    identify_model = IdentifySpeaker()

    identified_speakers = identify_speaker(tmp_path, identify_model, N_SPEAKERS)

    return identified_speakers


# Change a saved speaker id
@app.post("/change_speaker_id")
async def change_speaker_id(old_id: int, new_id: int):
    global collection

    output = collection.get(where={"speaker_id": str(old_id)})
    collection.update(ids=output["ids"], embeddings=output["embeddings"], metadatas=[{"speaker_id": new_id, "name": output["metadatas"][i]["name"], "details": output["metadatas"][i]["details"]} for i in range(0, len(output["ids"]))])

    return f"{len(output['ids'])} audio files updated"


# Change saved speaker details e.g. name
@app.post("/change_speaker_details")
async def change_speaker_details(id: int, new_id: int, name: str, details: str):
    global collection

    output = collection.get(where={"speaker_id": str(id)})

    if len(output["ids"]) == 0:
        return f"speaker {id} not found"

    collection.update(ids=output["ids"], embeddings=output["embeddings"], metadatas=[{"speaker_id": new_id, "name": name, "details": details} for i in range(0, len(output["ids"]))])

    return {f"speaker {new_id} saved",
            f"{len(output['ids'])} speaker ids changed"}


# Get speaker details by either name or id
@app.post("/get_speaker_details", description="search for saved speaker details")
async def search_for_speaker(search_value: str, search_field: str = Query('speaker_id', enum=['speaker_id', 'audio_id', 'name'])):
    global collection

    if search_field == "audio_id":
        output = collection.get(ids=[search_value])

    else:
        output = collection.get(where={search_field: search_value})

    if len(output["ids"]) == 0:
        return f"speaker {search_value} not found"

    result = {
        "audio_ids": output["ids"],
        "number_of_files": len(output["ids"]),
        "speaker_id": output["metadatas"][0]["speaker_id"],
        "speaker_name": output["metadatas"][0]["name"],
        "details": output["metadatas"][0]["details"]
    }

    return result


# Detect if there is a speaker present (y/n)
@app.post("/detect_speakers", description="Detect if speaker is present in audio")
def detect(audio_file: UploadFile):
    global diarise_model

    tmp_path = save_upload_file_tmp(audio_file)

    return diarise_model.detect_speakers(tmp_path)


# Save speakers to disk
@app.get("/persist", description="Persist speaker data")
def persist():
    global client

    client.persist()

    return "speakers saved"


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()

    return tmp_path
