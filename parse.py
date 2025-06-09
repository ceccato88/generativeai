# parse.py
"""
Baixa um PDF, faz o upload para o ChatDOC, aguarda o processamento
e baixa os dados estruturados em formato JSON.

Requisitos:
- pip install requests python-dotenv
- Coloque CHATDOC_API_KEY=<seu_token> no arquivo .env ou no ambiente.
"""

from dotenv import load_dotenv
import os
import sys
import io
import time
import requests
import json

# Carrega variáveis do .env no início do script
load_dotenv()

# -------------- CONFIGURAÇÃO --------------
API_KEY = os.getenv("CHATDOC_API_KEY")
PDF_URL = "https://arxiv.org/pdf/2505.20368.pdf"
PACKAGE = "elite"     # "elite" ou "basic"
OCR_MODE = "disable"  # "disable" | "auto" | "force"
BASE_URL = "https://api.chatdoc.com/api/v2"

# Constantes da API para clareza
STATUS_PROCESSING_COMPLETE = 300
POLLING_INTERVAL_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 120
# -----------------------------------------

if not API_KEY:
    sys.exit("❗ Defina CHATDOC_API_KEY em uma variável de ambiente ou no arquivo .env")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def download_pdf(url: str) -> io.BytesIO | None:
    """Baixa um arquivo PDF de uma URL e o retorna como um stream de bytes."""
    print(f"⬇️  Baixando PDF de: {url}...")
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()  # Lança uma exceção para status de erro (4xx ou 5xx)
        print("✅ PDF baixado com sucesso.")
        return io.BytesIO(r.content)
    except requests.exceptions.RequestException as e:
        print(f"🚫 Erro ao baixar o PDF: {e}")
        return None


def upload_to_chatdoc(pdf_stream: io.BytesIO, filename: str) -> str | None:
    """Faz o upload do PDF para o ChatDOC e retorna o ID do upload."""
    pdf_stream.seek(0)
    files = {
        "file": (filename, pdf_stream, "application/pdf"),
        "package_type": (None, PACKAGE),
        "ocr": (None, OCR_MODE),
    }
    upload_url = f"{BASE_URL}/documents/upload"

    print("⬆️  Enviando para o ChatDOC...")
    try:
        r = requests.post(upload_url, headers=HEADERS, files=files, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()

        data = r.json().get("data", {})
        upload_id = data.get("id")
        if not upload_id:
            print(f"🚫 Não foi possível obter o ID do upload da resposta: {r.json()}")
            return None

        print(f"✅ Upload iniciado com sucesso! ID: {upload_id}")
        return upload_id
    except requests.exceptions.RequestException as e:
        print(f"🚫 Erro na requisição de upload: {e}")
        return None


def wait_for_processing(upload_id: str) -> bool:
    """Verifica o status do processamento do documento até que esteja concluído ou falhe."""
    status_url = f"{BASE_URL}/documents/{upload_id}"
    print("⏳ Aguardando o processamento do documento...")

    while True:
        try:
            r = requests.get(status_url, headers=HEADERS, timeout=30)
            r.raise_for_status()

            status = r.json().get("data", {}).get("status")

            if status == STATUS_PROCESSING_COMPLETE:
                print("🎉 Documento processado com sucesso!")
                return True

            if status is None or status < 0:
                print(f"⚠️ Erro de processamento (status {status}). Resposta da API: {r.json()}")
                return False

            print(f"   Status atual: {status}. Aguardando {POLLING_INTERVAL_SECONDS} segundos...")
            time.sleep(POLLING_INTERVAL_SECONDS)

        except requests.exceptions.RequestException as e:
            print(f"🚫 Erro ao verificar o status: {e}")
            return False


def fetch_and_save_data(upload_id: str) -> None:
    """Busca os dados JSON processados e os salva em um arquivo."""
    data_url = f"{BASE_URL}/pdf_parser/{upload_id}"
    print(f"📄 Buscando dados estruturados para o ID: {upload_id}")

    try:
        r = requests.get(data_url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()

        pdf_data = r.json()
        output_file = f"{upload_id}_data.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=4)

        print(f"💾 Dados salvos com sucesso em: {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"🚫 Erro na requisição para obter os dados: {e}")


def main():
    """Orquestra todo o processo de ponta a ponta."""
    print("--- Iniciando processo de parse do PDF ---")

    pdf_buffer = download_pdf(PDF_URL)
    if not pdf_buffer:
        return

    filename = os.path.basename(PDF_URL) or "upload.pdf"

    upload_id = upload_to_chatdoc(pdf_buffer, filename)
    if not upload_id:
        return

    if wait_for_processing(upload_id):
        fetch_and_save_data(upload_id)

    print("--- Processo finalizado ---")


if __name__ == "__main__":
    main()