"""
Tarea 6
Autores:    Laura Sofía Garza Villarreal 600650
            Rafael Romero Hurtado        628911
Contacto:   laura.garzav@udem.edu
            rafael.romero@udem.edu
Organización: Universidad de Monterrey
Fecha de entrega: 22/03/2024
"""

# Importar librerias
import cv2 
import argparse
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union, List


# Define and initialise global variables
""" HSV_params almacena los valores para realizar la conversión al espacio de 
color HSV. """
HSV_params = {#'low_H': 0,        # Matriz
              #'low_H' : 58,
              #'low_H' : 82,
              'low_H' : 102,
              'high_H': 180,     # Matriz
              'low_S': 0,        # Saturacion
              #'high_S': 255,     # Saturacion
              'high_S' : 63,
              'low_V': 0,        # Brillo
              #'high_V': 255      # Brillo
              'high_V' : 126
            }

""" window_params almacena los nombres de las ventanas para mostra las imágenes
capturadas y las imágenes con los objetos en movimiento detectados"""
window_params = {'capture_window_name':'Input video',
                 'detection_window_name':'Detected object'}

""" text_params almacena los textos que se mostraran en los trackbars para 
los diferentes valores de HSV. """
text_params = {'low_H_text': 'Low H',
               'low_S_text': 'Low_S',
               'low_V_text': 'Low V',
               'high_H_text': 'High H',
               'high_S_text': 'High S',
               'high_V_text': 'High V'}

""" Esta función se ejecutará en el momento en que el trachbar de low_H se ajuste,
haciendo que se actualice este valor en HSV_params, así como también se asegura de
que se encuentre del rango válido y actualiza la posición del trackbar. """
def on_low_H_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['low_H'] = val
    HSV_params['low_H'] = min(HSV_params['high_H']-1, HSV_params['low_H'])
    cv2.setTrackbarPos(text_params['low_H_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['low_H'])

""" Esta función se ejecutará en el momento en que el trachbar de high_H se ajuste,
haciendo que se actualice este valor en HSV_params, así como también se asegura de
que se encuentre del rango válido y actualiza la posición del trackbar. """
def on_high_H_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['high_H'] = val
    HSV_params['high_H'] = max(HSV_params['high_H'], HSV_params['low_H']+1)
    cv2.setTrackbarPos(text_params['high_H_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['high_H'])

""" Esta función se ejecutará en el momento en que el trachbar de low_S se ajuste,
haciendo que se actualice este valor en HSV_params, así como también se asegura de
que se encuentre del rango válido y actualiza la posición del trackbar. """
def on_low_S_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['low_S'] = val
    HSV_params['low_S'] = min(HSV_params['high_S']-1, HSV_params['low_S'])
    cv2.setTrackbarPos(text_params['low_S_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['low_S'])

""" Esta función se ejecutará en el momento en que el trachbar de high_S se ajuste,
haciendo que se actualice este valor en HSV_params, así como también se asegura de
que se encuentre del rango válido y actualiza la posición del trackbar. """
def on_high_S_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['high_S'] = val
    HSV_params['high_S'] = max(HSV_params['high_S'], HSV_params['low_S'] +1)
    cv2.setTrackbarPos(text_params['high_S_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['high_S'])

""" Esta función se ejecutará en el momento en que el trachbar de low_V se ajuste,
haciendo que se actualice este valor en HSV_params, así como también se asegura de
que se encuentre del rango válido y actualiza la posición del trackbar. """
def on_low_V_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['low_V'] = val
    HSV_params['low_V'] = min(HSV_params['high_V']-1, HSV_params['low_V'])
    cv2.setTrackbarPos(text_params['low_V_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['low_V'])

""" Esta función se ejecutará en el momento en que el trachbar de high_V se ajuste,
haciendo que se actualice este valor en HSV_params, así como también se asegura de
que se encuentre del rango válido y actualiza la posición del trackbar. """
def on_high_V_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['high_V'] = val
    HSV_params['high_V'] = max(HSV_params['high_V'], HSV_params['low_V']+1)
    cv2.setTrackbarPos(text_params['high_V_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['high_V'])


def parse_cli_data()->argparse:
    parser = argparse.ArgumentParser(description='Tunning HSV bands for object detection')
    parser.add_argument('--video_file', 
                        type=str, 
                        default='camera', 
                        help='Video file used for the object detection process')
    parser.add_argument('--frame_resize_percentage', 
                        type=int, 
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    args = parser.parse_args()

    return args

""" Esta función toma un argumento dado por el usuario en la línea de comando, en este
caso el video que se analizara y realiza una captura de video y lo guarda en cap. """
def initialise_camera(args:argparse)->cv2.VideoCapture:
    
    # Create a video capture object
    cap = cv2.VideoCapture(args.video_file)
    
    return cap

""" En esta función se crean dos ventanas, una con el nombre de 'capture_window_name', y 
la otra con el nombre de 'detection_window_name', en donde en la primera se mostrará
la imagen capturada y en la segunda los objetos detectados. De igual manera se crean las
barras de seguimiento (trackbars) para cada parámetro en la ventana de detección para que se ajusten los 
valores del HSV. """
def configure_trackbars()->None:

    # Create two new windows for visualisation purposes 
    cv2.namedWindow(window_params['capture_window_name'])
    cv2.namedWindow(window_params['detection_window_name'])

    # Configure trackbars for the low and hight HSV values
    cv2.createTrackbar(text_params['low_H_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['low_H'], 
                       180, 
                       on_low_H_thresh_trackbar)
    cv2.createTrackbar(text_params['high_H_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['high_H'], 
                       180, 
                       on_high_H_thresh_trackbar)
    cv2.createTrackbar(text_params['low_S_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['low_S'], 
                       255, 
                       on_low_S_thresh_trackbar)
    cv2.createTrackbar(text_params['high_S_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['high_S'], 
                       255, 
                       on_high_S_thresh_trackbar)
    cv2.createTrackbar(text_params['low_V_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['low_V'], 
                       255, 
                       on_low_V_thresh_trackbar)
    cv2.createTrackbar(text_params['high_V_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['high_V'], 
                       255, 
                       on_high_V_thresh_trackbar)

""" En esta función se toma una imagen y un porcentaje que el usuario ingresa y devuelve la imagen
redimensionada."""
def rescale_frame(frame:NDArray, percentage:np.intc=20)->NDArray:
    
    # Resize current frame
    # Se calcula el nuevo ancho y alto de la imagen a partir del porcentaje dado
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    # cv2.resize redimensiona la imagen con interpolación y la devuelve en la variable frame
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

def segment_object(cap:cv2.VideoCapture, args:argparse)->None:

    # Variable para llevar la cuenta del número de fotograma procesado
    frame_number = 0

    # Main loop, estara activo mientras la cámara este abierta
    while cap.isOpened():

        # Incrementar el contador del número de fotograma 1
        frame_number += 1

        # Lee el fotograma actual, en donde ret se encarga de indicar si la lectura
        # fue exitosa y frame contiene el fotograma 
        ret, frame = cap.read()

        # Verifica que la imagen se haya capturado correctamente, de no ser así imprime
        # un mensaje de error y finaliza el bucle
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        # Redimensiona el fotograma actual con la función rescale_frame que se 
        # explicó anteriormente
        frame = rescale_frame(frame, args.frame_resize_percentage)
        
        # Applica un filtro de meriana al fotograma actual para eliminar el ruido y suavizarlo
        frame = cv2.medianBlur(frame,5)

        # Convierte el fotograma actual de BGR a HSV para que sea más fácil la 
        # segmentación de los objetos basada en el color
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Aplica umbral al fotograma HSV para poder identificar las áreas de interés a partir de
        # los valores de matriz, saturación y brillo proporconados anteriormente
        frame_threshold = cv2.inRange(frame_HSV, 
                                      (HSV_params['low_H'], 
                                       HSV_params['low_S'], 
                                       HSV_params['low_V']), 
                                      (HSV_params['high_H'], 
                                       HSV_params['high_S'], 
                                       HSV_params['high_V']))

        # Filtra la zona en donde hay cesped del fotograma actual manteniendo el objeto en movimiento
        bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)

        # Encuentra los contornos en la imagen con el umbral de HSV con la función findContours de OpenCV
        contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Revisa si se encontraron los contornos
        if contours:
            # Encuentra el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)

            # Da el rectangulo que delimita al objeto en movimiento alrededor del contorno más grande
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Dibuja el rectangulo en el fotograma original con la función rectangle de OpenCV
            cv2.rectangle(frame, (x-10, y-10), (x + w, y + h), (0, 255, 0), 0)

            # Agregar número de fotograma al fotograma original usando la función putText de OpenCV
            cv2.putText(frame, f'Frame: {frame_number}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Visualiza el video de entrada y el de detección de objetos con la función imshow de OpenCV
        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], bitwise_AND)

        # El programa finaliza si la tecla q o ESC se presionan y se imprime un mensaje de que
        # finalizo el programa y finaliza el blucle
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break

""" En esta función se cierran todas las ventanas que se hayan creado liberando la memoria que 
este asociada con esas ventanas, y libera los objetos o recursos utilizados por VideoCapture haciendo
que se libere la camara """
def close_windows(cap:cv2.VideoCapture)->None:
    
    # Cierra todas las ventanas creadas
    cv2.destroyAllWindows()

    # Cierra los objetos de 'VideoCapture'
    cap.release()