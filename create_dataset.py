import os
import pickle
import mediapipe as mp
import cv2

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path ke folder data
DATA_DIR = './data'

# Tempat menyimpan data
data = []
labels = []

# Cek apakah folder data ada
if not os.path.exists(DATA_DIR):
    print(f"❌ Folder '{DATA_DIR}' tidak ditemukan.")
    exit()

# Loop setiap subfolder di dalam ./data
for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        continue  # Lewati kalau bukan folder

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Cek apakah file adalah gambar
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        data_aux = []
        x_ = []
        y_ = []

        # Baca gambar
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Gagal membaca: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"⚠️ Tidak ada tangan terdeteksi di: {img_path}")

# Simpan hasilnya kalau ada data
if data:
    output_path = os.path.join(os.getcwd(), 'data.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"✅ {len(data)} data berhasil disimpan ke '{output_path}'")
else:
    print("❌ Tidak ada data yang berhasil diproses. Periksa gambar dan deteksi tangan.")
