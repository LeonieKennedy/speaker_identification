# Libraries
import pickle
from fastapi import FastAPI, UploadFile, Path, Query
from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import split_speakers
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
PERSIST_ENCODING_MULTIPLE = 3
count = 0

# Load saved encodings
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


@app.post("/upload_audio")
async def upload_audio_file(audio_file: UploadFile):
    global diarise_model, count

    model = SplitSpeakers()
    tmp_path = save_upload_file_tmp(audio_file)
    count, result = model.split_and_save_audio(tmp_path, count)

    return result

@app.post("/change_speaker_id")
async def change_speaker_id(old_id: int, new_id: int):
    global saved_speaker_ids, saved_speaker_details

    count = 0
    for i in range(0, len(saved_speaker_ids)):
        if saved_speaker_ids[i] == old_id:
            saved_speaker_ids[i] = new_id
            count += 1

    del saved_speaker_details[old_id]

    return f"{count} speakers updated"


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


@app.post("/identify_speaker", description="Identify single speaker in audio file")
async def identify_single_speaker(audio_file: UploadFile):
    global saved_speaker_embeddings, saved_speaker_ids, saved_speaker_details

    print("identify speaker")
    tmp_path = save_upload_file_tmp(audio_file)

    identify_model = IdentifySpeaker(saved_speaker_ids, saved_speaker_embeddings)

    embedding = identify_model.create_embedding(tmp_path)
    speaker_id = len(saved_speaker_details)

    if identify_model.linearsvc is None:
        identified_speakers = f"no speakers saved - saving new speaker with id {speaker_id}"
    else:
        identified_speakers = identify_model.identify_speaker(embedding, saved_speaker_details)
        identified_speakers.append(("saved_id " + str(speaker_id)))

    saved_speaker_embeddings.append(embedding)
    saved_speaker_ids.append(speaker_id)
    saved_speaker_details[speaker_id] = {
        "name": ("unknown_" + str(speaker_id)),
        "details": ""
    }

    return identified_speakers


@app.post("/detect_speakers", description="Detect if speaker is present in audio")
def detect(audio_file: UploadFile):
    global diarise_model

    tmp_path = save_upload_file_tmp(audio_file)

    return diarise_model.detect_speakers(tmp_path)


@app.post("/persist", description="Persist speaker data")
def persist():
    global saved_speaker_embeddings, saved_speaker_details, saved_speaker_ids

    pickle.dump(saved_speaker_embeddings, open("saved_speakers/saved_encodings.pt", "wb"))
    pickle.dump(saved_speaker_details, open("saved_speakers/saved_speaker_details.pt", "wb"))
    pickle.dump(saved_speaker_ids, open("saved_speakers/saved_speaker_ids.pt", "wb"))

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
