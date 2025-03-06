import os
from subprocess import run

def generar_inpainting_lama_cleaner(ruta_imagen_entrada, ruta_mascara, ruta_imagen_salida):
    """
    Genera un inpainting utilizando Lama Cleaner a través de la línea de comandos.

    Este script asume que tienes Lama Cleaner instalado y accesible desde la línea de comandos.
    Para usarlo, debes tener una imagen y una máscara que indique las áreas a rellenar.

    Args:
        ruta_imagen_entrada (str): Ruta al archivo de imagen de entrada.
        ruta_mascara (str): Ruta al archivo de imagen de máscara.
                              Las áreas blancas en la máscara indican las regiones a inpintar.
        ruta_imagen_salida (str): Ruta donde se guardará la imagen con el inpainting.
    """

    comando = [
        "lama-cleaner",  # Comando para ejecutar Lama Cleaner (puede variar según tu instalación)
        "--image", ruta_imagen_entrada,
        "--mask", ruta_mascara,
        "--output", ruta_imagen_salida
    ]

    try:
        resultado = run(comando, capture_output=True, text=True, check=True)
        print("Inpainting completado con éxito.")
        if resultado.stdout:
            print("Salida del proceso:\n", resultado.stdout)
        if resultado.stderr:
            print("Errores (si los hay):\n", resultado.stderr)

    except FileNotFoundError:
        print("Error: El comando 'lama-cleaner' no se encontró. Asegúrate de que Lama Cleaner esté instalado y en tu PATH.")
        print("Consulta la documentación de Lama Cleaner para la instalación y uso en línea de comandos.")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar Lama Cleaner. Código de retorno: {e.returncode}")
        print("Salida de error de Lama Cleaner:\n", e.stderr)
        print("Por favor, revisa los mensajes de error para diagnosticar el problema.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    # Ejemplo de uso:
    imagen_entrada = "img1.jpg"  # Reemplaza con la ruta a tu imagen
    mascara = "img2.jpg"   # Reemplaza con la ruta a tu máscara
    imagen_salida = "img3.png"    # Ruta donde se guardará la imagen resultante

    # Asegúrate de que los archivos de imagen y máscara existan
    if not os.path.exists(imagen_entrada):
        print(f"Error: No se encontró la imagen de entrada en: {imagen_entrada}")
    elif not os.path.exists(mascara):
        print(f"Error: No se encontró la máscara en: {mascara}")
    else:
        generar_inpainting_lama_cleaner(imagen_entrada, mascara, imagen_salida)
        print(f"Imagen inpintada guardada en: {imagen_salida}")