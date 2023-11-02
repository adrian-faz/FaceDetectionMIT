import cv2
import face_recognition
from pathlib import Path
import pickle
from collections import Counter
import datetime
import shutil

# Pasos para instalar el programa:

# Para Mac:
# brew update
# brew install cmake gcc

# Para Windows:
# choco install mingw

# Crear un Virtual Environment
# Correr el siguiente comando para instalar los paquetes y dependencias:
# python -m pip install -r requirements.txt

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

def get_person_name():
    name = input("Enter the name of the person: ")
    return name


# Realizamos el encoding de las caras con las imagenes
def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def recognize_faces_in_image(
    image_path: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Recognize faces in a given image.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Load the image and detect faces
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=model)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Iterate through faces and try to recognize
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = _recognize_face(face_encoding, loaded_encodings)
        if not name:
            name = "Unknown"

        # Draw bounding box and label on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Convert the image to BGR color (which OpenCV uses)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Show the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("Welcome")
# Enco
recognize_faces_in_image("validation/3face10.jpg")
