# Libraries
import pickle

from fastapi import FastAPI, UploadFile, Path
from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

# Scripts
import save_encodings
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
PERSIST_ENCODING_MULTIPLE = 1000
count = 0

diarise_model = SplitSpeakers()
identify_model = IdentifySpeaker()

# Load saved encodings
try:
    encodings = pickle.load(open("saved_speakers.pt", "rb"))
except FileNotFoundError:
    encodings = []


@app.post("/upload_audio", tags=["Diarise audio file"])
async def upload_audio_file(audio_file: UploadFile):
    global diarise_model, identify_model

    return


@app.post("/detect_speakers", tags=["Detect if speaker is present in audio"])
async def detect(audio_file: UploadFile):
    global diarise_model

    tmp_path = save_upload_file_tmp(audio_file)

    return SplitSpeakers.detect_speakers(diarise_model, tmp_path)


@app.post("/persist", tags=["Persist speaker data"])
async def persist():
    global encodings

    save_encodings.persist(encodings)

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
