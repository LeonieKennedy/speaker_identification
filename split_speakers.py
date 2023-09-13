import numpy as np
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os


# For diarisation and speaker detection
class SplitSpeakers:
    def __init__(self):
        print("creating models")

        # If you wanted to run it online
        voice_token = "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"
        pipeline_token = "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"

        self.voice = Pipeline.from_pretrained("./models/voice_config.yaml")
        self.pipeline = Pipeline.from_pretrained("./models/diarisation_config.yaml")
        print("model, pipeline and inference created")

    def split_and_save_audio(self, audio_file, count):
        audio = AudioSegment.from_wav(audio_file)

        diarisation = self.pipeline(audio_file)
        count += 1
        other_count = 0
        new_speaker_dict = []
        for time, _, speaker in diarisation.itertracks(yield_label=True):
            speaker = (speaker[-2:])

            audio_segment = audio[(time.start * 1000):(time.end * 1000)]

            try:
                audio_segment.export("./audio_files/"+speaker + "_" + str(count) + "/" + speaker + "_" + str(other_count) + ".wav", format="wav")
            except FileNotFoundError:
                os.makedirs("./audio_files/" + speaker + "_" + str(count))
                audio_segment.export("./audio_files/"+speaker + "_" + str(count) + "/" + speaker + "_" + str(other_count) + ".wav", format="wav")
                new_speaker_dict.append(speaker)
            other_count += 1

        return count, (f"{len(np.unique(new_speaker_dict))} speakers found. {len(diarisation)} files saved in ./audio_files/")

    # Detect if there are voices present in file
    def detect_speakers(self, audio):
        # Output is a list of speakers and times
        output = self.voice(audio)
        print(output)

        if len(output) > 0:
            return "speakers likely present"
        else:
            return "no speakers detected"
