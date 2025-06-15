import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model berhasil dimuat.")
except Exception as e:
    print("Gagal memuat model:", e)
    exit()

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # Ubah ke 0 untuk default webcam
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label klasifikasi
labels_dict = {
    0: 'Gaya (F) = m x a',                      # Hukum Newton 2
    1: 'Gravitasi (Fg) = G x (m₁ x m₂) / rxr',    # Gaya gravitasi universal
    2: 'Lentur = (M x y) / I',               # Tegangan lentur
    3: 'Listrik (V) = I x R',                    # Hukum Ohm
    4: 'Magnet (F) = q x v x B'                  # Gaya magnetik (Lorentz force)
}

print("Program dimulai. Tekan 'q' untuk keluar.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Tampilkan prediksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        except Exception as e:
            print("Gagal melakukan prediksi:", e)

    # Tampilkan frame
    cv2.imshow('Sign Language Detector', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Keluar dari program.")
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
