import torch
import torchaudio


class IdentifySpeaker:
    def __init__(self):
        self.wav2mel = torch.jit.load("wav2mel.pt")
        self.dvector = torch.jit.load("dvector.pt").eval()

    # Create embedding for a single audio file that contains a single speaker
    def create_embedding(self, path):
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

        return embedding

