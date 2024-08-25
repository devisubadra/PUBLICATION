import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical

# Load Haar Cascade for face detection
def load_face_cascade():
    """
    Load the pre-trained Haar Cascade for face detection.
    
    Returns:
        face_cascade (cv2.CascadeClassifier): Haar Cascade classifier.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Detect faces in an image
def detect_faces(image, face_cascade):
    """
    Detect faces in an image using Haar Cascade.
    
    Args:
        image (np.array): Input image.
        face_cascade (cv2.CascadeClassifier): Haar Cascade classifier.
    
    Returns:
        faces (list of tuples): List of face bounding boxes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Align faces in the image
def align_faces(image, faces):
    """
    Extract and resize faces from the image.
    
    Args:
        image (np.array): Input image.
        faces (list of tuples): List of face bounding boxes.
    
    Returns:
        aligned_faces (list of np.array): List of aligned faces.
    """
    aligned_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))  # Resize to model input size
        aligned_faces.append(face)
    return aligned_faces

# Load and preprocess data
def load_data(image_paths, labels):
    """
    Load images and their corresponding labels.
    
    Args:
        image_paths (list of str): List of file paths to the images.
        labels (list of str): List of labels corresponding to the images.

    Returns:
        X (np.array): Array of image data.
        y (np.array): Array of labels.
    """
    X = []
    y = []

    for image_path, label in zip(image_paths, labels):
        img = cv2.imread(image_path)
        faces = detect_faces(img, load_face_cascade())
        aligned_faces = align_faces(img, faces)
        
        for face in aligned_faces:
            X.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))  # Convert to grayscale
            y.append(label)
    
    X = np.array(X).reshape(-1, 64, 64, 1)  # Add channel dimension
    y = np.array(y)
    return X, y

# Define emotion classification model
def build_model():
    """
    Build a CNN model for emotion classification.
    
    Returns:
        model (keras.models.Sequential): Compiled Keras model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # Adjust based on the number of emotion classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train the emotion classification model
def train_model(X_train, y_train, X_val, y_val):
    """
    Train the emotion classification model.
    
    Args:
        X_train (np.array): Training feature data.
        y_train (np.array): Training labels.
        X_val (np.array): Validation feature data.
        y_val (np.array): Validation labels.

    Returns:
        model (keras.models.Sequential): Trained Keras model.
    """
    model = build_model()

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_val, y_val),
              epochs=25,
              verbose=1)
    
    return model

# Predict and evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on the test set.
    
    Args:
        model (keras.models.Sequential): Trained Keras model.
        X_test (np.array): Test feature data.
        y_test (np.array): Test labels.
    
    Returns:
        None
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    print(f'Accuracy: {model.evaluate(X_test, y_test)[1] * 100:.2f}%')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Real-time emotion prediction from webcam feed
def real_time_emotion_detection(model):
    """
    Real-time emotion detection using a webcam feed.
    
    Args:
        model (keras.models.Sequential): Trained Keras model.
    
    Returns:
        None
    """
    face_cascade = load_face_cascade()
    vs = VideoStream(src=0).start()

    while True:
        frame = vs.read()
        faces = detect_faces(frame, face_cascade)
        aligned_faces = align_faces(frame, faces)
        
        for face in aligned_faces:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray = np.expand_dims(face_gray, axis=0)
            face_gray = np.expand_dims(face_gray, axis=-1)
            face_gray = face_gray / 255.0  # Normalize

            prediction = model.predict(face_gray)
            emotion = np.argmax(prediction)

            # Display the emotion on the frame
            (x, y, w, h) = faces[0]
            cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

# Main function
def main():
    """
    Main function to execute the workflow: load data, train model, evaluate model, and real-time detection.
    
    Returns:
        None
    """
    # Example image paths and labels (replace with your actual dataset)
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
    labels = ['happy', 'sad', ...]  # Replace with actual labels

    # Load and preprocess data
    X, y = load_data(image_paths, labels)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)  # One-hot encode labels

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Real-time emotion detection from webcam feed
    real_time_emotion_detection(model)

if __name__ == "__main__":
    main()
