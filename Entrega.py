import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def contour_and_extract_chromosomes(image_path):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Aplicar un umbral para binarizar la imagen
    _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Eliminar objetos pequeños para reducir ruido
    binary_image = remove_small_objects(binary_image > 0, min_size=100).astype(np.uint8) * 255

    # Calcular la transformada de distancia
    distance = ndi.distance_transform_edt(binary_image)

    # Detectar máximos locales en la misma forma que `binary_image`
    local_maxi = peak_local_max(distance, footprint=np.ones((20, 20)), labels=binary_image)
    markers = np.zeros_like(binary_image, dtype=int)
    markers[tuple(local_maxi.T)] = np.arange(1, local_maxi.shape[0] + 1)

    # Aplicar watershed para segmentar los cromosomas
    labels = watershed(-distance, markers, mask=binary_image)

    # Convertir la imagen original a color para visualizar los contornos en color
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Dibujar y extraer contornos de cada cromosoma
    contours, _ = cv2.findContours((labels > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_chromosomes = []
    overlap_threshold = 500  # Umbral para cromosomas sobrepuestos
    damage_min_threshold = 70  # Umbral mínimo de área para considerar cromosomas dañados
    damage_max_threshold = 800  # Umbral máximo de área para considerar cromosomas dañados
    circularity_threshold = 0.5  # Circularidad mínima para cromosomas no dañados
    aspect_ratio_threshold = 3.0  # Relación de aspecto máxima para cromosomas no dañados

    for i, contour in enumerate(contours):
        # Encontrar el rectángulo delimitador de cada contorno
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calcular el área del contorno
        area = cv2.contourArea(contour)
        
        # Calcular perímetro y circularidad (4π * área / perímetro^2)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0
        
        # Calcular relación de aspecto
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        # Determinar el tipo de cromosoma basado en el área, circularidad y relación de aspecto
        if area > overlap_threshold:
            # Contornear en amarillo si es sobrepuesto
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 255), 2)  # Color amarillo
        elif area < damage_min_threshold or area > damage_max_threshold or circularity < circularity_threshold or aspect_ratio > aspect_ratio_threshold:
            # Contornear en rojo si es dañado (muy pequeño/grande, baja circularidad o aspecto inusual)
            cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 255), 2)  # Color rojo
        else:
            # Contornear en verde si es cromosoma normal
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)  # Color verde

        # Recortar el cromosoma de la imagen original
        chromosome_img = image[y:y+h, x:x+w]

        # Verificar la forma de la imagen recortada
        print(f"Shape of cropped chromosome {i}: {chromosome_img.shape}")

        # Asegurarse de que la imagen recortada tiene la forma correcta (2D)
        if len(chromosome_img.shape) == 2 and chromosome_img.shape[0] > 1 and chromosome_img.shape[1] > 1:
            cropped_chromosomes.append(chromosome_img)
        else:
            print(f"Skipping invalid cropped chromosome {i} with shape {chromosome_img.shape}")

    # Mostrar la imagen original con los contornos
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("Contornos de Cromosomas")
    plt.axis("off")

    # Mostrar los cromosomas recortados
    plt.subplot(1, 2, 2)
    if cropped_chromosomes:
        cols = 5  # Columnas en la cuadrícula de visualización
        rows = (len(cropped_chromosomes) + cols - 1) // cols  # Filas necesarias
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        for ax, chromo_img in zip(axes.flat, cropped_chromosomes):
            ax.imshow(chromo_img, cmap="gray")
            ax.axis("off")
        plt.suptitle("Cromosomas Recortados")
    else:
        plt.text(0.5, 0.5, 'No se detectaron cromosomas recortados', ha='center', va='center')
    plt.axis("off")
    plt.show()


# Ejemplo de uso
contour_and_extract_chromosomes("C:/Users/isark/Desktop/Vision Computacional/originals/original 3.bmp")
