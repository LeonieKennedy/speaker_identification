import pickle


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


def persist(encodings):
    pickle.dump(encodings, open("saved_speakeres.pt", "wb"))
