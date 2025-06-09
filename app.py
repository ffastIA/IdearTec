import os  # Importa o módulo 'os' para interagir com o sistema operacional, como acessar variáveis de ambiente.
import textwrap
from dotenv import load_dotenv  # Importa a função 'load_dotenv' do pacote 'dotenv' para carregar variáveis de ambiente de um arquivo .env.
from langchain_core.prompts import ChatPromptTemplate  # Importa 'ChatPromptTemplate' da LangChain para criar templates de prompts para modelos de chat.
from langchain.chains.combine_documents import create_stuff_documents_chain  # Importa a função para criar uma cadeia que "recheia" (stuff) documentos no prompt.
from langchain_community.document_loaders.pdf import PyPDFLoader  # Importa 'PyPDFLoader' da LangChain para carregar documentos PDF.
from langchain_openai import ChatOpenAI  # Importa 'ChatOpenAI' para interagir com os modelos de chat da OpenAI.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importa 'RecursiveCharacterTextSplitter' para dividir textos grandes em pedaços menores.
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
import tiktoken
import streamlit as st
from langchain_community.vectorstores import FAISS



def cria_vector_store_faiss(chunks: List[Document]):
    """cria a vector store"""
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings_model
    )
    vectorstore.save_local(diretorio_vectorestore_faiss)
    return vectorstore

def carrega_vector_store_faiss(diretorio_vectorestore_faiss, embeddings_model):
    vectorstore = FAISS.load_local(diretorio_vectorestore_faiss, embeddings_model)
    return vectorstore


# 1. Defina a função de contagem de tokens usando tiktoken
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Retorna o número de tokens no texto de string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# --- Configuração de Variáveis de Ambiente e Chave da API ---
load_dotenv()  # Carrega as variáveis de ambiente do arquivo .env (se existir) para o ambiente do sistema.
openai_key = os.getenv(
    'OPENAI_API_KEY')  # Tenta obter a chave da API da OpenAI da variável de ambiente 'OPENAI_API_KEY'.

# openai_api_key = st.secrets["OPENAI_API_KEY"]

# Verifica se a chave da API foi carregada com sucesso.
if not openai_key:  # Se a chave não for encontrada, levanta um erro para alertar o usuário.
    raise ValueError("A variável de ambiente 'OPENAI_API_KEY' não foi encontrada no seu arquivo .env.")

#modelo de llm selecionado
modelo='gpt-3.5-turbo-0125'

#modelo de embedding selecionado
embeddings_model = OpenAIEmbeddings()

#diretório onde será criada a vectore store
diretorio_vectorestore_faiss = 'vectorestore_faiss'


# --- Carregamento do Documento PDF ---
# Define o caminho para o arquivo PDF no sistema de arquivos.
caminho_arquivo = r'BIA_RAG.pdf'

print(f"Carregando documentos do PDF: {caminho_arquivo}")  # Informa o usuário sobre o carregamento do PDF.
try:  # Inicia um bloco try-except para lidar com possíveis erros durante o carregamento do arquivo.
    loader = PyPDFLoader(caminho_arquivo)  # Instancia um PyPDFLoader, passando o caminho do arquivo PDF.
    documentos = loader.load()  # Carrega o conteúdo do PDF. Cada página do PDF se torna um 'Document' separado na lista 'documentos'.
    print(
        f"PDF carregado. Total de documentos (páginas): {len(documentos)}")  # Confirma o carregamento e exibe o número de páginas/documentos.
except FileNotFoundError:  # Captura o erro específico se o arquivo PDF não for encontrado.
    print(
        f"Erro: O arquivo PDF não foi encontrado em '{caminho_arquivo}'. Por favor, verifique o caminho.")  # Informa o erro ao usuário.
    exit()  # Interrompe a execução do script se o arquivo não for encontrado, pois é uma dependência crítica.

# --- Depuração e Inspeção de Documentos (Opcional) ---
# print("\n--- Conteúdo da primeira página para verificação ---")
# print(
#     documentos[0].page_content[:500] + "...")  # Imprime os primeiros 500 caracteres da primeira página para depuração.
# print("\n--- Metadados da primeira página ---")
# print(documentos[0].metadata)  # Imprime os metadados associados à primeira página.
# print("-" * 50 + "\n")

# --- Configuração do Text Splitter (para futura implementação de RAG avançado) ---
# O text splitter foi configurado aqui, mas note que no laço principal abaixo,
# o contexto ainda usa 'documentos[:10]' e não os 'chunks' criados por este splitter.
# Para utilizar os chunks, seria necessário adaptar a forma como o 'context' é passado para 'chain.invoke'.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Tamanho máximo do chunk em tokens (usando a função de contagem do módulo 'tokens').
    chunk_overlap=50,  # Sobreposição em tokens entre chunks.
    length_function= num_tokens_from_string,  # Nossa função personalizada de contagem de tokens.
    separators= ['&', '\n\n','.', ' '],
    add_start_index=True  # Adiciona o índice de início de cada chunk no texto original como metadado.
)

# Divide o documento completo em chunks menores.
# Esta parte do código processa o PDF inteiro e o divide, mas não é usada na invocação da chain neste exemplo,
# que ainda opera sobre 'documentos[:10]'.
print("Processando o PDF em chunks para contagem de tokens (apenas para demonstração de split):")
chunks = text_splitter.split_documents(documentos)
print(f"Texto original dividido em {len(chunks)} chunks.\n")

# Comentado para evitar uma saída muito longa, mas pode ser descomentado para depuração.
# for i, chunk in enumerate(chunks):
#     content = chunk.page_content
#     token_count = tokens.num_tokens_from_string(content)
#     print(f"--- Chunk {i+1} (Início: {chunk.metadata['start_index']}, Tokens: {token_count}) ---")
#     print(content)
#     print("-" * 50 + "\n")

# --- Inicialização do Modelo de Linguagem (LLM) ---
chat = ChatOpenAI(
    model=modelo,  # Especifica o modelo da OpenAI a ser usado .
    # openai_api_key: openai_key,  # Fornece a chave da API para autenticação.
    temperature=0
    # Define a "temperatura" do modelo. Um valor de 0 torna as respostas mais determinísticas e menos criativas/aleatórias.
)  # Cria uma instância do modelo de chat da OpenAI.

# --- Criação do Prompt Template ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     ("Você é um assistente especialista no contexto fornecido. Use o seguinte contexto para responder à pergunta."
     "Os tópicos principais estão destacado entre aspas duplas. Se a resposta não" 
     " estiver no contexto, diga que não sabe e peça mais detalhes para o questionamento:\n\n{context}")),
    # Define a mensagem do sistema, instruindo o modelo sobre seu papel e comportamento.
    # '{context}' é um placeholder para o conteúdo dos documentos.
    ("user", "{question}")
    # Define a mensagem do usuário, que conterá a pergunta real.
    # '{question}' é um placeholder para a pergunta do usuário.
])  # Constrói um template de prompt que será usado para formatar as entradas para o LLM.

# --- Criação da Cadeia de Combinação de Documentos ---
chain = create_stuff_documents_chain(llm=chat, prompt=qa_prompt)
# Cria uma "chain" que pega os documentos e os "recheia" (stuff) diretamente no placeholder '{context}' do 'qa_prompt'.
# Esta estratégia é adequada quando o tamanho total dos documentos (ou páginas selecionadas) cabe no limite de tokens do LLM.

# --- Loop Principal para Interação com o Usuário ---
while True:  # Inicia um loop infinito que só será interrompido pela escolha do usuário.
    print("\n--- Menu de Opções ---")
    print("1. Criar vector store")
    print("2. Carrega vector store")
    print("3. Fazer uma pergunta ao documento")
    print("4. Sair")

    # vectorstore = FAISS.load_local(diretorio_vectorestore_faiss, embeddings_model, allow_dangerous_deserialization=True)
    # print(vectorstore.index.ntotal)
    # print("Vector Store carregado com sucesso!!!")
    # st.write(vectorstore.index.ntotal)
    # st.write("Vector Store carregado com sucesso!!!")


# st.write("\n--- Menu de Opções ---")
# # st.write("2. Carrega base de dados")
# st.write("1. Fazer uma pergunta sobre o projeto")
# st.write("2. Limpar")
# # st.write("3. Criar base de dados")

    escolha = input("Digite sua opção (1 , 2 , 3 ou 4): ").strip()  # Captura a escolha do usuário e remove espaços em branco.

# # Usa session_state para evitar recriação com mesma key
# if "escolha" not in st.session_state:
#     st.session_state["escolha"] = 2
#
# escolha = st.number_input("Escolha uma opção:", min_value=1, max_value=2, value=2)
# st.session_state["escolha"] = escolha

    if escolha == '1':
        vectorstore = cria_vector_store_faiss(chunks)
        num_chunks = len(vectorstore.index_to_docstore_id)
        print(f"Número de chunks no FAISS: {num_chunks}")
        print("Vector Store criado com sucesso!!!")
#     # st.write(f"Número de chunks no FAISS: {num_chunks}")
#     # st.write("Vector Store criado com sucesso!!!")

    if escolha == '2':
        # st.session_state["escolha"] = escolha
        vectorstore = FAISS.load_local(diretorio_vectorestore_faiss, embeddings_model, allow_dangerous_deserialization=True)
        print(vectorstore.index.ntotal)
        print("Vector Store carregado com sucesso!!!")

    elif escolha == '3':
        # pergunta = input("Digite sua pergunta:")
        pergunta = st.text_input("Digite sua pergunta: ")
        vectorstore = FAISS.load_local(diretorio_vectorestore_faiss, embeddings_model, allow_dangerous_deserialization=True)
        chat = ChatOpenAI(model=modelo)
        chat_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_type='mmr'),
            return_source_documents=True
        )
        resposta_do_chat = chat_chain.invoke({'query': pergunta})
        resposta_llm = resposta_do_chat.get('result', 'Nenhuma resposta disponível.')
        resposta_formatada = textwrap.fill(resposta_llm, width=120)
        print(resposta_formatada)
        # st.write(resposta_formatada)

    elif escolha == '4':
        st.write("Saindo do programa. Até mais!")  # Mensagem de despedida.


    else:
        print("Opção inválida. Por favor, digite 1 ou 2.")  # Informa o usuário sobre uma entrada inválida.
        # st.write("Opção inválida. Por favor, digite 1 ou 2.")  # Informa o usuário sobre uma entrada inválida.
