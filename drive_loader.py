import io

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_DOC_EXPORT_MIME = "text/plain"
GOOGLE_SHEET_EXPORT_MIME = "text/csv"


def _get_drive_service(service_account_file):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    return build("drive", "v3", credentials=credentials)


SUPPORTED_MIME_TYPES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/markdown",
}


def _extract_text(service, file_id, mime_type):
    if mime_type == "application/vnd.google-apps.document":
        data = service.files().export(fileId=file_id, mimeType=GOOGLE_DOC_EXPORT_MIME).execute()
        return data.decode("utf-8")

    if mime_type == "application/vnd.google-apps.spreadsheet":
        data = service.files().export(fileId=file_id, mimeType=GOOGLE_SHEET_EXPORT_MIME).execute()
        return data.decode("utf-8")

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buffer.seek(0)

    if mime_type == "application/pdf":
        reader = PdfReader(buffer)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return buffer.read().decode("utf-8", errors="ignore")


def fetch_drive_file_text(file_id, service_account_file):
    """Downloads a Drive file and returns (name, extracted_text).

    Handles Google Docs/Sheets (via export) and binary files like PDF or
    plain text (via direct download).
    """
    service = _get_drive_service(service_account_file)
    meta = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    text = _extract_text(service, file_id, meta["mimeType"])
    return meta["name"], text


def fetch_drive_folder_texts(folder_id, service_account_file):
    """Returns a list of (name, text) for every supported file directly
    inside the given Drive folder. Unsupported file types are skipped.
    """
    service = _get_drive_service(service_account_file)
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=1000,
    ).execute()

    documents = []
    for file in results.get("files", []):
        if file["mimeType"] not in SUPPORTED_MIME_TYPES:
            print(f"Skipping '{file['name']}' (unsupported type: {file['mimeType']})")
            continue
        text = _extract_text(service, file["id"], file["mimeType"])
        documents.append((file["name"], text))

    return documents
