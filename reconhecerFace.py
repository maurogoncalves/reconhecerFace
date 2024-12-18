# Importar bibliotecas
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar o modelo FaceNet
# baixar o modelo
facenet_model = load_model('facenet_keras.h5')

# Inicializar o detector MTCNN
detector = MTCNN()

# Função para extrair embeddings faciais
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    face_pixels = np.expand_dims(face_pixels, axis=0)
    return model.predict(face_pixels)[0]

# Carregar imagem
image = cv2.imread('suaFoto.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar rostos
faces = detector.detect_faces(image_rgb)

# Processar e exibir rostos detectados
for face in faces:
    x, y, width, height = face['box']
    face_image = image_rgb[y:y+height, x:x+width]
    face_image = cv2.resize(face_image, (160, 160))

    # Gerar embedding
    embedding = get_embedding(facenet_model, face_image)
    print("Embedding gerado com sucesso:", embedding[:5])  # Exibir os primeiros 5 valores do vetor

    # Desenhar bounding box
    cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (255, 0, 0), 2)

# Mostrar resultado
plt.imshow(image_rgb)
plt.axis('off')
plt.show()