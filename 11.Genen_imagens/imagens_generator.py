import streamlit as st
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

# Configuração básica da página do Streamlit
st.set_page_config(page_title="IA Generativa", layout="wide")
st.title("Gerador de Imagens com Stable Diffusion")


def generate_images(
    prompt: str,
    negative_prompt: str,
    num_images_per_prompt: int,
    num_inference_steps: int,
    height: int,
    width: int,
    seed: int,
    guidance_scale: float,
) -> list:
    """
    Gera imagens usando o modelo de difusão estável.

    Parâmetros:
    - prompt (str): O prompt para a geração das imagens.
    - negative_prompt (str): O prompt negativo para a geração das imagens.
    - num_images_per_prompt (int): O número de imagens a serem geradas por prompt.
    - num_inference_steps (int): O número de passos de inferência a serem executados.
    - height (int): A altura das imagens geradas.
    - width (int): A largura das imagens geradas.
    - seed (int): A semente para a geração de números aleatórios.
    - guidance_scale (float): A escala de orientação para a geração das imagens.

    Retorna:
    - images (list): Uma lista de imagens geradas.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "stabilityai/stable-diffusion-2-1-base"

    # Configura o scheduler e o pipeline
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, scheduler=scheduler
    ).to(device)

    # Define o gerador com a semente especificada
    generator = torch.Generator(device=device).manual_seed(seed)

    # Gera as imagens
    images = pipeline(
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=guidance_scale,
    )["images"]

    return images


# Configurações de interface do usuário na barra lateral do Streamlit
with st.sidebar:
    st.header("Configurações da Geração da Imagem")
    prompt = st.text_area("Prompt", "")
    negative_prompt = st.text_area("Negative Prompt", "")
    num_images_per_prompt = st.slider(
        "Número de Imagens", min_value=1, max_value=5, value=1
    )
    num_inference_steps = st.number_input(
        "Número de Passos de Inferência", min_value=1, max_value=100, value=50
    )
    height = st.selectbox("Altura da Imagem", [256, 512, 768, 1024], index=1)
    width = st.selectbox("Largura da Imagem", [256, 512, 768, 1024], index=1)
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=42)
    guidance_scale = st.number_input(
        "Escala de Orientação", min_value=1.0, max_value=20.0, value=7.5
    )
    generate_button = st.button("Gerar Imagem")

if generate_button and prompt:
    with st.spinner("Gerando Imagens..."):
        images = generate_images(
            prompt,
            negative_prompt,
            num_images_per_prompt,
            num_inference_steps,
            height,
            width,
            seed,
            guidance_scale,
        )
        cols = st.columns(len(images))
        for idx, (col, img) in enumerate(zip(cols, images)):
            with col:
                st.image(
                    img,
                    caption=f"Imagem {idx + 1}",
                    use_column_width=True,
                    output_format="auto",
                )
