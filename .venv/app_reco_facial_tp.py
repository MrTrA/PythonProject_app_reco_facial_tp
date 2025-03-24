import cv2
import streamlit as st
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_faces() :
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    while True :
        # Lire les images de la webcam
        ret, frame = cap.read()
        if not ret:
          print("Erreur lors de la capture d'image")
          return
        # Convertit les images en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Détecter les visages à l'aide du classificateur de cascade de visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Dessine des rectangles autour des visages détectés
        for (x, y, w, h) in faces :
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Afficher les images
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Sortir de la boucle lorsque 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Libère la webcam et ferme toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()
def app() :
    st.title("Détection de visage à l'aide de l'algorithme de Viola-Jones" )
    st.write("Appuyez sur le bouton ci-dessous pour commencer à détecter des visages à partir de votre webcam")
    # Ajouter un bouton pour commencer à détecter les visages
    if st.button("Détecter les visages" ):
        # Appeler la fonction detect_faces
        detect_faces()
if __name__ == "__main__" :
    app()