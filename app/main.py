import os

import streamlit as st

from converter import convert_pdf_to_markdown, convert_docx_to_markdown

st.set_page_config(page_title="Konwerter do Markdown", layout="wide")

st.title("Konwerter PDF / DOCX do Markdown")
st.write("Przeciagnij pliki PDF lub DOCX, a otrzymasz wierny Markdown (konwersja 1:1, bez modyfikacji tresci).")

uploaded_files = st.file_uploader(
    "Wybierz pliki do konwersji",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Przeciagnij lub wybierz pliki PDF / DOCX aby rozpoczac konwersje.")
    st.stop()

for uploaded_file in uploaded_files:
    st.divider()
    st.subheader(uploaded_file.name)

    file_bytes = uploaded_file.read()
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    try:
        if extension == "pdf":
            status_placeholder = st.empty()
            def update_status(msg):
                status_placeholder.info(msg)
            with st.spinner(f"Konwersja {uploaded_file.name}..."):
                md_result = convert_pdf_to_markdown(file_bytes, on_status=update_status)
            status_placeholder.empty()
        elif extension == "docx":
            md_result = convert_docx_to_markdown(file_bytes)
        else:
            st.error(f"Nieobslugiwany format: .{extension}")
            continue
    except Exception as e:
        st.error(f"Blad konwersji pliku {uploaded_file.name}: {e}")
        continue

    # Auto-save
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    md_filename = uploaded_file.name.rsplit(".", 1)[0] + ".md"
    output_path = os.path.join(outputs_dir, md_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_result)
    st.success(f"Zapisano: outputs/{md_filename}")

    tab_source, tab_preview = st.tabs(["Zrodlo Markdown", "Podglad renderowany"])

    with tab_source:
        st.code(md_result, language="markdown")

    with tab_preview:
        st.markdown(md_result, unsafe_allow_html=False)

    st.download_button(
        label=f"Pobierz {md_filename}",
        data=md_result,
        file_name=md_filename,
        mime="text/markdown",
        key=f"download_{uploaded_file.name}",
    )
