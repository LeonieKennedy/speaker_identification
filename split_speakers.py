import numpy as np
from pyannote.audio import Model, Pipeline, Inference
from pydub import AudioSegment
from identify_speaker import IdentifySpeaker
import os

# For diarisation and speaker detection
class SplitSpeakers:
    def __init__(self):
        print("creating models")
        voice_embedding_token = "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"
        pipeline_inference_token = "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"

        self.voice = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=voice_embedding_token)
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=pipeline_inference_token)
        print("model, pipeline and inference created")


    def split_and_save_audio(self, audio_file, count):
        audio = AudioSegment.from_wav(audio_file)

        diarisation = self.pipeline(audio_file)
        count += 1
        new_speaker_dict = []
        for time, _, speaker in diarisation.itertracks(yield_label=True):
            speaker = (speaker[-2:])

            audio_segment = audio[time.start:time.end]

            try:
                audio_segment.export("./audio_files/"+speaker + "_" + str(count) + "/" + speaker + "_"+str(time.start)[3:] + ".wav", format="wav")
            except FileNotFoundError:
                os.makedirs("./audio_files/" + speaker + "_" + str(count))
                audio_segment.export("./audio_files/"+speaker + "_" + str(count) + "/" + speaker + "_"+str(time.start)[3:] + ".wav", format="wav")
                new_speaker_dict.append(speaker)

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
