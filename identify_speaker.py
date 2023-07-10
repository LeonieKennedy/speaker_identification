import datetime
import traceback

import numpy as np
import torch
import torchaudio
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


class IdentifySpeaker:
    def __init__(self, speakers, embeddings):
        print("init")
        start = datetime.datetime.now()

        self.wav2mel = torch.jit.load("models/wav2mel.pt")
        self.dvector = torch.jit.load("models/dvector.pt").eval()

        unique_speaker_id = np.unique(speakers)
        no_embeddings = len(embeddings)

        print(no_embeddings)
        class_weights = {}
        for id in unique_speaker_id:
            class_weights[id] = (no_embeddings) / (len(unique_speaker_id) * speakers.count(id))

        print(class_weights)
        try:
            self.random_forest = RandomForestClassifier(class_weight=class_weights).fit(embeddings, speakers)
            self.gaussian = GaussianNB().fit(embeddings, speakers)
            self.kneighbours = KNeighborsClassifier(weights='distance').fit(embeddings, speakers)

            # Network error here
            old = LinearSVC(class_weight=class_weights)
            print("here")
            self.linearsvc = old.fit(embeddings, speakers)
            print("past")
        except ValueError:
            self.random_forest = None
            self.gaussian = None
            self.kneighbours = None
            self.linearsvc = None
            print(f"ValueError: \n {traceback.format_exc()}")
        except:
            print(f"Error in linearsvc: \n {traceback.format_exc()}")

        end = datetime.datetime.now()
        print("Here")
        print("time:", end - start)
        print("embeddings:", no_embeddings)
        print("classes:", len(unique_speaker_id))

    # Create embedding for a single audio file that contains a single speaker
    def create_embedding(self, path):
        try:
            print("creating tensor")
            # Create mel spectrogram - pre-processes audio file to normalise volume, remove silence etc
            wav_tensor, sample_rate = torchaudio.load(path)
            mel_tensor = self.wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
            print("tensor created")
            # Create the embedding
            emb = self.dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
            embedding = emb.detach().numpy()

        # Occasionally you get a RuntimeError when audio file is too short (<1s)
        except RuntimeError:
            print(f"Path {path} too short. RuntimeError")
            return "RuntimeError"
        print("embedding created")
        return embedding

    # Identify speaker using 4 different classifier models
    #   - KNeighbours: highest class count within a circumference
    #   - GaussianNB: Bayes Theorem
    #   - LinearSVC: splits data with line.
    #   -  RandomForest: multiple decision trees
    def identify_speaker(self, embedding, speaker_details):
        embedding = embedding.reshape(1, -1)
        results = []
        models = ["KNeighbours", "GuassianNB", "RandomForest"]
        count = 0
        for model in [self.kneighbours, self.gaussian, self.random_forest]:
            try:
                prediction = model._predict_proba_lr(embedding)[0].tolist()
            except AttributeError:
                prediction = model.predict_proba(embedding)[0].tolist()

            print("len1", len(speaker_details))
            print("len2", len(prediction))
            prediction_count = 0
            for i in speaker_details:
                print(prediction_count)
                print(prediction[prediction_count])
                print(speaker_details[i]["name"], prediction[prediction_count])
                prediction_count += 1

            id = prediction.index(max(prediction))

            results.append({
                "model": models[count],
                "speaker_id": id,
                "speaker_name": speaker_details[list(speaker_details)[id]]["name"],
                "confidence": max(prediction)
            })

            count += 1

        return results
