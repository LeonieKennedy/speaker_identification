import traceback

import torch
import torchaudio



class IdentifySpeaker:
    def __init__(self):
        self.wav2mel = torch.jit.load("models/wav2mel.pt")
        self.dvector = torch.jit.load("models/dvector.pt").eval()

    # Create embedding for a single audio file that contains a single speaker
    def create_embedding(self, path):
        print("create embedding")
        try:
            # Create mel spectrogram - pre-processes audio file to normalise volume, remove silence etc
            wav_tensor, sample_rate = torchaudio.load(path)
            mel_tensor = self.wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)

            # Create the embedding
            emb = self.dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
            embedding = emb.detach().numpy()

        # Occasionally you get a RuntimeError when audio file is too short (<1s)
        except RuntimeError:
            print(f"Path {path} too short. RuntimeError")
            traceback.print_exc()
            return "RuntimeError"

        print("embedding created")

        return embedding

    # Identify speaker
    def identify_speaker(self, embedding, collection, multiple_speakers, n_results):
        embedding = embedding.reshape(1, -1).tolist()

        print("identify count:", collection.count())

        output = collection.query(
            query_embeddings=embedding,
            n_results=n_results
        )
        print("output:\n", output)

        results = []
        for i in range(0, len(output["ids"][0])):
            results.append({
                "speaker_id": output["metadatas"][0][i]["speaker_id"],
                "name": output["metadatas"][0][i]["name"],
                "details": output["metadatas"][0][i]["details"],
                "distance": output["distances"][0][i]
            })

        print(multiple_speakers)
        if multiple_speakers is False:
            collection.add(embeddings=embedding,
                           metadatas=[{"speaker_id": str(collection.count() + 1), "name": ("Unknown" + str(collection.count() + 1)), "details": ""}],
                           ids=str(collection.count() + 1))

        return results
