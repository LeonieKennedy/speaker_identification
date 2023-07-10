# Libraries
import pickle

import uvicorn
from fastapi import FastAPI, UploadFile, Path, Query
from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

# Scripts
from split_speakers import SplitSpeakers
from identify_speaker import IdentifySpeaker

# Set up API
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

# Load saved speakers
try:
    saved_speaker_embeddings = pickle.load(open("saved_speakers/saved_speaker_embeddings.pkl", "rb"))
    saved_speaker_details = pickle.load(open("saved_speakers/saved_speaker_details.pkl", "rb"))
    saved_speaker_ids = pickle.load(open("saved_speakers/saved_speaker_ids.pkl", "rb"))
    print(saved_speaker_details)
except FileNotFoundError:
    saved_speaker_embeddings = []
    saved_speaker_details = []
    saved_speaker_ids = []

# Models
diarise_model = SplitSpeakers()


def identify_speaker(file_path, model):
    global count, saved_speaker_embeddings, saved_speaker_details, saved_speaker_ids

    embedding = model.create_embedding(file_path)
    if embedding == "RuntimeError":
        return embedding
    speaker_id = len(saved_speaker_details)

    if model.kneighbours is None:
        identified_speakers = f"no speakers saved - saving new speaker with id {speaker_id}"
    else:
        identified_speakers = model.identify_speaker(embedding, saved_speaker_details)
        identified_speakers.append(("saved_id " + str(speaker_id)))

    saved_speaker_embeddings.append(embedding)
    saved_speaker_ids.append(speaker_id)
    saved_speaker_details[speaker_id] = {
        "name": ("unknown_" + str(speaker_id)),
        "details": ""
    }

    count += 1
    if count % PERSIST_ENCODING_MULTIPLE == 0:
        persist()
        print("saved to disk")

    return identified_speakers


# Upload audio with multiple speakers
@app.post("/upload_audio")
async def upload_audio_file(audio_file: UploadFile, save_and_identify: bool = False):
    global diarise_model, count, saved_speaker_embeddings, saved_speaker_ids

    tmp_path = save_upload_file_tmp(audio_file)
    print("diarising audio")
    count, result = diarise_model.split_and_save_audio(tmp_path, count)

    if save_and_identify is True:
        print("identifying speakers")
        identify_model = IdentifySpeaker(saved_speaker_ids, saved_speaker_embeddings)
        list_of_speakers = {}
        all_results = {}
        for directory in os.listdir("audio_files"):
            print(directory)
            for file in os.listdir(os.path.join("audio_files", directory)):
                file_path = os.path.join("audio_files", directory, file)
                identified_speakers = identify_speaker(file_path, identify_model)
                if identified_speakers == "RuntimeError":
                    continue
                print(identified_speakers)
                for speaker in identified_speakers:
                    try:
                        list_of_speakers[directory][speaker["speaker_id"]]["count"] += 1
                    except ValueError:
                        list_of_speakers[directory][speaker["speaker_id"]]["count"] = 1

            all_results[directory]["speaker_id"] = max(list_of_speakers[directory], key=list_of_speakers[directory].get)
            all_results[directory]["speaker_name"] = saved_speaker_details[all_results[directory]["speaker_id"]]["speaker_name"]
            print(all_results)
        return all_results

    return result


# Identify a speaker in an audio file with a single speaker in it
@app.post("/identify_speaker", description="Identify single speaker in audio file")
async def identify_single_speaker(audio_file: UploadFile):
    global saved_speaker_embeddings, saved_speaker_ids, saved_speaker_details, count

    print(type(audio_file))
    tmp_path = save_upload_file_tmp(audio_file)

    identify_model = IdentifySpeaker(saved_speaker_ids, saved_speaker_embeddings)

    identified_speakers = identify_speaker(tmp_path, identify_model)

    return identified_speakers


# Change a saved speaker id
@app.post("/change_speaker_id")
async def change_speaker_id(old_id: int, new_id: int):
    global saved_speaker_ids, saved_speaker_details

    no_changed_speakers = 0

    for i in range(0, len(saved_speaker_ids)):
        if saved_speaker_ids[i] == old_id:
            saved_speaker_ids[i] = new_id
            no_changed_speakers += 1

    del saved_speaker_details[old_id]

    return f"{count} speakers updated"


# Change saved speaker details e.g. name
@app.post("/change_speaker_details")
async def change_speaker_details(id: int, new_id: int, name: str, details: str):
    global saved_speaker_details, saved_speaker_ids
    print(saved_speaker_ids)
    try:
        count = 0
        if id != new_id:
            del saved_speaker_details[id]
            deleted = True
            for i in range(0, len(saved_speaker_ids)):
                print(saved_speaker_ids[i])
                if saved_speaker_ids[i] == id:
                    saved_speaker_ids[i] = new_id
                    count += 1
        else:
            deleted = False

        saved_speaker_details[new_id]["name"] = name
        saved_speaker_details[new_id]["details"] = details

        return {f"speaker {new_id} saved",
                f"{count} audio ids changed",
                f"speaker {id} deleted: {deleted}"}

    except KeyError:
        return f"speaker {id} not found"


# Get speaker details by either name or id
@app.post("/get_speaker_details", description="search for saved speaker details")
async def search_for_speaker(search_value: str, search_field: str = Query('id', enum=['id', 'name'])):
    global saved_speaker_details

    if search_field == "id":
        try:
            result = saved_speaker_details[int(search_value)]
            result["id"] = int(search_value)
            return result
        except:
            return f"speaker {search_value} not found"
    else:
        for i in range(0, len(saved_speaker_details)):
            if saved_speaker_details[i]["name"] == search_value:
                result = saved_speaker_details[i]
                result["id"] = i
                return result
        return f"no speaker called {search_value} found"


# Detect if there is a speaker present (y/n)
@app.post("/detect_speakers", description="Detect if speaker is present in audio")
def detect(audio_file: UploadFile):
    global diarise_model

    tmp_path = save_upload_file_tmp(audio_file)

    return diarise_model.detect_speakers(tmp_path)


# Save speakers to disk
@app.post("/persist", description="Persist speaker data")
def persist():
    global saved_speaker_embeddings, saved_speaker_details, saved_speaker_ids

    pickle.dump(saved_speaker_embeddings, open("saved_speakers/saved_encodings.pt", "wb"))
    pickle.dump(saved_speaker_details, open("saved_speakers/saved_speaker_details.pt", "wb"))
    pickle.dump(saved_speaker_ids, open("saved_speakers/saved_speaker_ids.pt", "wb"))

    return "speakers saved"


@app.get("/health")
async def health():
    return "alive"


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()

    return tmp_path

#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8002)
