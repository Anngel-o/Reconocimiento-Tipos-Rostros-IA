# Face Shape Recognition System

Sistema de reconocimiento de formas de rostro utilizando redes neuronales convolucionales (CNN) con interfaz gráfica en Python.

![Demo Interface](https://github.com/user-attachments/assets/900f2e4c-672b-4ea6-8565-ea0c7ba8b455)

## Características principales
- Clasificación de formas de rostro en 5 categorías: Corazón, Ovalada, Oblonga, Redonda y Cuadrada
- Interfaz gráfica intuitiva con Tkinter
- Muestra resultados con porcentaje de confianza
- Soporta formatos JPG, JPEG y PNG

## Requisitos del sistema
- Python 3.12 o superior
- pip (gestor de paquetes Python)

## Anexos
Dataset a Utilizar: https://www.kaggle.com/datasets/niten19/face-shape-dataset/data

Red Neuronal Entrenada: https://colab.research.google.com/drive/1xAFoak3Ahf3f_gEDJbF6YZrgITDEN0Xm?usp=sharing

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/face-shape-recognition.git
cd face-shape-recognition

# 2. Crear entorno virtual
**Windows:**
cmd
python -m venv venv
.\venv\Scripts\activate

**macOS/Linux:**
bash
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install tensorflow pillow numpy opencv-python tk

# 4. Ejecución
python interfaz3.py
