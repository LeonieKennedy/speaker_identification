import pickle


# Add speaker to list of speakers(dictionary)
def add_speaker(name, new_encoding, details, saved_encodings):
    speaker = {
        "name": name,
        "details": details,
        "encoding": new_encoding,
    }
    if saved_encodings is []:
        saved_encodings = [speaker]
    else:
        saved_encodings.append(speaker)

    return saved_encodings


# Save all speakers to disk
def persist(encodings, filename):
    pickle.dump(encodings, open(filename, "wb"))
