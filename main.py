import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas
 
# Si necesitas remover fondo para algo adicional:
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
 
@st.cache_resource
def load_ldm_inpainting_pipeline():
    """
    Carga el pipeline de Stable Diffusion Inpainting una sola vez.
    """
    from diffusers import StableDiffusionInpaintPipeline
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")  # Cambia a "cpu" si no tienes GPU
    return pipe_inpaint
 
def inpaint_image(pipe, image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Realiza el inpainting con parámetros fijos y muestra una barra de progreso:
    - 20 pasos
    - guidance_scale = 15
    - callback para progreso
    """
    prompt = "A clean scene without the selected object"
    negative_prompt = "blurry, distorted, mutated, disfigured, watermark, text, bad quality"
 
    orig_width, orig_height = image.size
    # Ajustar dimensiones a múltiplos de 8
    new_width = orig_width if orig_width % 8 == 0 else ((orig_width // 8) + 1) * 8
    new_height = orig_height if orig_height % 8 == 0 else ((orig_height // 8) + 1) * 8
 
    # Crear imágenes "paddeadas"
    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    padded_mask = Image.new("L", (new_width, new_height), 0)
    padded_image.paste(image, (0, 0))
    padded_mask.paste(mask, (0, 0))
 
    num_inference_steps = 20
    guidance_scale = 15
 
    # Barra de progreso
    progress_bar = st.progress(0)
 
    # Función callback para actualizar la barra
    def progress_callback(step, timestep, latents):
        percent = int((step + 1) / num_inference_steps * 100)
        progress_bar.progress(percent)
 
    with st.spinner("Procesando inpainting..."):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=padded_image,
            mask_image=padded_mask,
            height=new_height,
            width=new_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback=progress_callback,
            callback_steps=1
        )
 
    # Recortar al tamaño original
    final_image = result.images[0].crop((0, 0, orig_width, orig_height))
    return final_image
 
def main():
    st.title("Eliminar objetos con Inpainting (Brush)")
 
    # Cargamos pipeline de inpainting
    pipe_inpaint = load_ldm_inpainting_pipeline()
 
    # Estados en session_state
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "previous_image" not in st.session_state:
        st.session_state.previous_image = None
    # Para reiniciar el canvas tras hacer Remove
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
 
    # Subir imagen
    uploaded_file = st.file_uploader("1) Sube tu imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Solo inicializamos la primera vez
        if st.session_state.original_image is None:
            st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image = st.session_state.original_image.copy()
            st.session_state.previous_image = st.session_state.original_image.copy()
 
        current_img = st.session_state.current_image
 
        # Slider para el tamaño del pincel
        brush_size = st.slider("Brush size", min_value=5, max_value=100, value=25)
 
        # Key único para reiniciar el lienzo tras cada inpainting
        canvas_key = f"canvas_mask_{st.session_state.canvas_key}"
 
        # Canvas con la imagen actual como background
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=brush_size,
            stroke_color="rgba(255, 0, 0, 0.5)",  # Rojo semitransparente (50%)
            background_image=current_img,
            update_streamlit=True,
            width=current_img.width,
            height=current_img.height,
            drawing_mode="freedraw",
            key=canvas_key,
        )
 
        # Botones en la misma fila
        col_remove, col_return, col_continue = st.columns([1, 1, 1])
 
        with col_remove:
            if st.button("Remove"):
                if canvas_result.image_data is not None:
                    # Extraer la máscara
                    drawing = canvas_result.image_data.astype(np.uint8)
                    if drawing.shape[2] == 4:
                        alpha = drawing[:, :, 3]
                        mask_array = np.zeros((current_img.height, current_img.width), dtype=np.uint8)
                        mask_array[alpha > 0] = 255
                    else:
                        gray = cv2.cvtColor(drawing, cv2.COLOR_RGB2GRAY)
                        _, mask_array = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
 
                    mask = Image.fromarray(mask_array)
                    # Guardar imagen anterior por si se quiere "Regresar"
                    st.session_state.previous_image = st.session_state.current_image.copy()
 
                    # Inpainting
                    st.session_state.current_image = inpaint_image(
                        pipe=pipe_inpaint,
                        image=current_img,
                        mask=mask
                    )
 
                    # Incrementamos la clave para reiniciar el lienzo
                    st.session_state.canvas_key += 1
                else:
                    st.warning("No has pintado nada en el canvas. Selecciona algo con el pincel antes de pulsar 'Remove'.")
 
                st.rerun()
 
        with col_return:
            if st.button("Regresar"):
                temp = st.session_state.current_image
                st.session_state.current_image = st.session_state.previous_image
                st.session_state.previous_image = temp
                st.rerun()
 
        with col_continue:
            if st.button("Continue"):
                st.write("Aquí puede continuar con otro paso (p.ej. insertar mueble, etc.)")
                # Implementa tu lógica de 'siguiente paso' o paginación:
                # st.session_state.page = "insertar_mueble"
                # st.experimental_rerun()
                # etc.
 
if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 481de658e8820fdc6d83804a1426649a62094b52
