import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Identificación de Forma de Rostro")
        self.root.geometry("800x500")
        self.root.configure(bg='white')
        
        # Cargar el modelo de TensorFlow
        try:
            self.model = load_model('face_shape_model(2).h5')
            self.class_names = ['Corazón', 'Ovalada', 'Oblonga', 'Redonda', 'Cuadrada']  # Ajusta según tus clases
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {str(e)}")
            self.root.destroy()
            return
        
        # Variable para almacenar la ruta de la imagen
        self.image_path = None
        
        # Crear el frame principal
        main_frame = tk.Frame(root, bg='white')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Primera columna - Imagen
        self.image_frame = tk.Frame(main_frame, bg='lightgray', width=300, height=400, relief='sunken', bd=2)
        self.image_frame.grid(row=0, column=0, padx=(0, 20), pady=0, sticky='nsew')
        self.image_frame.grid_propagate(False)
        
        # Label para mostrar la imagen
        self.image_label = tk.Label(self.image_frame, text="Imagen del rostro\n(No hay imagen cargada)", 
                                  bg='lightgray', fg='gray', font=('Arial', 12))
        self.image_label.pack(expand=True, fill='both')
        
        # Segunda columna - Controles
        controls_frame = tk.Frame(main_frame, bg='white')
        controls_frame.grid(row=0, column=1, sticky='nsew', padx=(20, 0))
        
        # Título
        title_label = tk.Label(controls_frame, text="Identificación de Forma de Rostro", 
                             font=('Arial', 18, 'bold'), bg='white', fg='black')
        title_label.pack(pady=(0, 20))
        
        # Texto descriptivo
        desc_label = tk.Label(controls_frame, 
                            text="Sube una foto frontal de un rostro y te diremos su forma geométrica", 
                            font=('Arial', 11), bg='white', fg='black', wraplength=300, justify='left')
        desc_label.pack(pady=(0, 30))
        
        # Frame para los botones
        buttons_frame = tk.Frame(controls_frame, bg='white')
        buttons_frame.pack(pady=(0, 30))
        
        # Botón "Subir foto"
        self.upload_btn = tk.Button(buttons_frame, text="Subir foto", 
                                  font=('Arial', 11), bg='#4CAF50', fg='white',
                                  padx=20, pady=10, command=self.upload_image)
        self.upload_btn.pack(side='left', padx=(0, 10))
        
        # Botón "Calcular"
        self.calculate_btn = tk.Button(buttons_frame, text="Identificar Forma", 
                                     font=('Arial', 11), bg='#2196F3', fg='white',
                                     padx=20, pady=10, command=self.predict_face_shape,
                                     state='disabled')
        self.calculate_btn.pack(side='left')
        
        # Texto "Resultado:"
        result_title = tk.Label(controls_frame, text="Resultado:", 
                              font=('Arial', 12, 'bold'), bg='white', fg='black')
        result_title.pack(pady=(0, 10), anchor='w')
        
        # Resultado
        self.result_label = tk.Label(controls_frame, text="---", 
                                   font=('Arial', 14), bg='white', fg='blue')
        self.result_label.pack(anchor='w')
        
        # Confianza
        self.confidence_label = tk.Label(controls_frame, text="Confianza: ---", 
                                       font=('Arial', 10), bg='white', fg='gray')
        self.confidence_label.pack(anchor='w')
        
        # Configurar el grid para que sea responsive
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
    
    def upload_image(self):
        """Función para subir una imagen"""
        file_types = [
            ('Archivos de imagen', '*.jpg *.jpeg *.png'),
            ('Todos los archivos', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title='Seleccionar imagen',
            filetypes=file_types
        )
        
        if filename:
            self.image_path = filename
            self.display_image(filename)
            self.calculate_btn.config(state='normal')
            self.result_label.config(text="---")
            self.confidence_label.config(text="Confianza: ---")
    
    def display_image(self, image_path):
        """Función para mostrar la imagen cargada"""
        try:
            # Cargar y redimensionar la imagen
            image = Image.open(image_path)
            
            # Redimensionar manteniendo la proporción
            image.thumbnail((280, 380), Image.Resampling.LANCZOS)
            
            # Convertir para tkinter
            photo = ImageTk.PhotoImage(image)
            
            # Mostrar en el label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Mantener una referencia
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocesa la imagen para el modelo"""
        try:
            # Cargar imagen
            img = Image.open(image_path)
            
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Redimensionar a 128x128 (tamaño que espera tu modelo)
            img = img.resize((128, 128))
            
            # Convertir a array y normalizar
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
            
            return img_array
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
            return None
    
    def predict_face_shape(self):
        """Función para predecir la forma del rostro"""
        if not self.image_path:
            messagebox.showwarning("Advertencia", "Por favor, primero sube una imagen.")
            return
        
        try:
            # Preprocesar imagen
            processed_img = self.preprocess_image(self.image_path)
            
            if processed_img is None:
                return
                
            # Hacer predicción
            predictions = self.model.predict(processed_img)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Mostrar resultados
            result_text = self.class_names[predicted_class]
            confidence_text = f"Confianza: {confidence*100:.1f}%"
            
            self.result_label.config(text=result_text)
            self.confidence_label.config(text=confidence_text)
            
            # Mostrar mensaje
            messagebox.showinfo(
                "Análisis completado",
                f"Forma de rostro identificada: {result_text}\n{confidence_text}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la predicción: {str(e)}")

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()