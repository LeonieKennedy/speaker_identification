# Speaker Identification

Identify and save speakers


## How to run
Run the following commands:
```commandline
sudo docker build -t speaker_identifier:latest .
sudo docker run --name speaker_identifier -p 8000:8000 --gpus all -d speaker_identifier:latest
```
Navigate to: http://0.0.0.0:8000/docs#/

## Endpoints
### Upload Audio
Upload an audio file that contains multiple speakers. The file will then be split up into smaller audio files and
grouped by speaker. They will be saved to a directory. 

If the option to identify the speakers is selected, then the speakers will be identified by looking at the most commonly
identified speaker across all the files in that speakers directory.

Returns the speaker id, name and confidence where anything between 0 - 0.5 should be considered high confidence.
Anything over 1 should be considered low confidence.

Speakers are not saved due to the diarisation not being too accurate.

### Identify Speaker
Upload an audio file that contains only one speaker. The audio will then be compared to other saved audio embeddings.

Returns the top 3 speakers with their speaker_id, name, details and confidence.

Submitted audio file is added to the collection of saved audio embeddings, but assigned a name of "Unknown"(+number)
and a new speaker_id, which can be changed once identity of speaker is confirmed.

### Change Speaker Id
Change the id of all audio embeddings for a saved speaker. Old speaker details and names are changed to the new speaker's
details and name, as well as the speaker_id.

### Change Speaker Details
Change the details for all files with the id inputted. If the id is the same as the new_id, then the speaker id will not
be changed.

### Get Speaker Details
Search for a speaker by name, speaker_id, or audio_id.

Returns audio_ids, information on the speaker, as well as the number of audio files submitted for that speaker.

### Delete speaker
Deletes an audio file or speaker by inputting the field and value. If you want to delete the speaker, submit speaker_id
as the field (or name if their name is unique), and the speaker_id as the value. If you want to delete a single audio
embedding, enter audio_id as the field and the audio_id of the embedding as the value.

### Detect speakers
Detect if a speaker is present in an audio file. If false, then there is not a speaker present. If true, there is likely
to be a speaker present, however it could also be picking up background noise.

### Persist
Save audio embeddings and speakers to disk. This function is called automatically every x times a speaker is saved to 
the collection, however you should run this function once you have submitted all of your audio files.