"""
Tarea 6
Autores:    Laura Sofía Garza Villarreal 600650
            Rafael Romero Hurtado        628911
Contacto:   laura.garzav@udem.edu
            rafael.romero@udem.edu
Organización: Universidad de Monterrey
Fecha de entrega: 22/03/2024
"""

# Import standard libraries 
import cv2 
import argparse
import numpy as np
import od                               # Importar módulo de python con funciones para la detección de movimiento
from numpy.typing import NDArray
from typing import Dict, Union, List

""" En esta función se toma el argumento del video que se analizará que brinda el usuario
en la línea de comandos"""
def run_pipeline(args:argparse)->None:

    # Inicializa la función initialise_camera que fue explicada en el módulo anterior
    cap = od.initialise_camera(args)

    # Configura los trackbars para los valores de HSV mas bajos y altos usando la 
    # función configura_trachbars explicada en el módulo anterior
    od.configure_trackbars()

    # Procesa el video con la función segment_object del módulo anterior
    od.segment_object(cap, args)

    # Cierra todas las ventanas abiertas creadas con la función close_windows 
    # del módulo anterior
    od.close_windows(cap)

# Función principal
if __name__=='__main__':

    # Obtiene los datos de la línea de comandos con la función parse_cli_data
    # del módulo od y lo almacena en args
    args = od.parse_cli_data()

    # Ejecuta la función run_pipline con los datos obtenidos de la línea de comados
    run_pipeline(args)