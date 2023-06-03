from pyannote.audio import Model, Pipeline, Inference
from pydub import AudioSegment

class SplitSpeakers:
    def __init__(self):
        embedding_token = "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"
        pipeline_inference_token = "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"

        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=embedding_token)
        # self.model.to("cuda:0")

        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=pipeline_inference_token)
        self.inference = Inference(self.model, window="whole")
        print("model, pipeline and inference created")

    def split_file(self, audio_file):
        pass
