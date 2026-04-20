from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from cyclegan_interiors import (
    CHECKPOINT_PATH,
    create_model,
    get_device,
    load_checkpoint,
    preprocess_image,
    tensor_to_pil,
)


EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


@st.cache_resource
def load_model():
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    device = get_device()
    model = create_model(device)
    load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()
    return model, device


def translate(image: Image.Image, direction: str) -> Image.Image:
    model, device = load_model()

    source_domain = "A" if direction == "modern -> rustic" else "B"
    target_domain = "B" if direction == "modern -> rustic" else "A"

    tensor = preprocess_image(image, source_domain).to(device)

    with torch.inference_mode():
        generator = model.generators["a_to_b"] if direction == "modern -> rustic" else model.generators["b_to_a"]
        translated = generator(tensor)

    return tensor_to_pil(translated, target_domain)


st.set_page_config(page_title="CycleGAN Interiors", page_icon="🏠", layout="wide")

st.title("CycleGAN для интерьеров")
st.write("демо переводит интерьер между доменами `modern` и `rustic`.")

if not Path(CHECKPOINT_PATH).exists():
    st.error("веса модели не найдены.")
    st.stop()

with st.spinner("загружаю модель..."):
    load_model()

direction = st.radio(
    "направление перевода",
    ["modern -> rustic", "rustic -> modern"],
    horizontal=True,
)

if direction == "modern -> rustic":
    example_paths = sorted(EXAMPLES_DIR.glob("modern_*.jpg"))
else:
    example_paths = sorted(EXAMPLES_DIR.glob("rustic_*.jpg"))

example_names = [path.name for path in example_paths]

if not example_names:
    st.error("примеры не найдены.")
else:
    selected_name = st.selectbox("выберите пример интерьера", example_names)
    image = Image.open(EXAMPLES_DIR / selected_name).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("исходное изображение")
        st.image(image, use_column_width=True)

    if st.button("запустить перевод"):
        try:
            with st.spinner("выполняю перевод..."):
                result = translate(image, direction)

            with col2:
                st.subheader("результат")
                st.image(result, use_column_width=True)
        except Exception as error:
            st.error(f"не удалось выполнить инференс: {error}")
