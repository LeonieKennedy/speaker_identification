# speaker_identification
This project uses saved audio to identify known speakers and to save unknown speakers for future identification.

It first splits the audio by speaker, and groups each speaker together. It then creates a tensor for that audio and compares it with the saved audio tensors.

Machine learning models are used to convert the audio to mel spectrograms, which are then converted to tensors. The spectograms highlight audio features.
