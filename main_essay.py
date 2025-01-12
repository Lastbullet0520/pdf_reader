# RetrievalQA 사용
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import fitz
import os
import streamlit as st
import tempfile
from PIL import Image
import io

# /.devcontainer/devcontainer.json 기준, 해당 파일이 streamlit에 업로드 됩니다.

st.set_page_config(layout="wide")
# 제목
st.title("AI 논술선생님")
uploaded_file = st.file_uploader("논술문제를 업로드 해주세요")
st.write("---")

# 논술자료 pdf 업로드
def pdf_to_images(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    pdf_images = []

    # Convert each page to an image
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        pdf_img = Image.open(io.BytesIO(pix.tobytes()))
        pdf_images.append(pdf_img)
    return pdf_images

# 구성요소를 좌,우로 분할
col1, col2 = st.columns(2)

# 왼쪽 구성요소
with col1:
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Convert PDF to images
        images = pdf_to_images("uploaded.pdf")

        st.write(f"Total pages: {len(images)}")

        # Display each image
        for i, img in enumerate(images):
            st.image(img, caption=f'Page {i + 1}', use_column_width=True)
        st.write("---")





def pdf_to_document(upload_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, upload_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(upload_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    new_pages = loader.load_and_split()
    return new_pages


# 오른쪽 구성요소
with col2:
    if uploaded_file is not None:
        try:
            pages = pdf_to_document(uploaded_file)

            # Split
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a tiny chunk size, just to show.
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_documents(pages)

            # Embedding
            embeddings_model = OpenAIEmbeddings()

            # load it into Chroma

            chromadb = Chroma.from_documents(
                texts,
                embeddings_model,
                collection_name='esg',
            )

            # Question
            st.header("답안을 작성해 주세요")
            question = st.text_input('답안 작성')

            if st.button('제출하기'):
                with st.spinner('열심히 채점 중입니다...'):
                    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm,
                        retriever=chromadb.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    # 프롬프트 작성. 추가 수정 필요
                    result = qa_chain({"query": "당신은 논술 선생님입니다." +
                                                "학생이 작성한 답안을 보고 피드백을 주어야 합니다.\n" +
                                                "출력 양식은 다음과 같습니다 :\n\n" +
                                                "AI 선생님의 논문 요약 결과(Title1)\n" +
                                                "논문 요약(Bold) : {업로드 되어있는 논문의 요약을 보여주세요.}\n" +
                                                "논문 키워드(Bold) : {업로드 되어있는 논문의 핵심 키워드를 알려주세요.}\n" +
                                                "피드백(Title2)\n" +
                                                "* 잘한 점(Bold) : {논술 문제 내 키워드를 사용한 요소를 알려주세요.}\n" +
                                                "* 아쉬운 점(Bold) : {논술 문제 내 키워드를 사용하지 못한 요소를 알려주세요.}\n" +
                                                "이렇게 작성하면 더 좋아요!(Title2)\n" +
                                                "{업로드 되어있는 논술문제의 규칙을 반드시 지키되,공백을 제외한 글자수로 지켜주세요." +
                                                " 키워드를 사용한 모범 답변을 작성해주세요.}\n" +
                                                "아래는 학생이 작성한 답안입니다.\n\n" +
                                                question})
                    st.write(result["result"])

        except Exception as e:
            st.error(f"An error occurred: {e}")
