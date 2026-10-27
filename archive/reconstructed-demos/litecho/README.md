# Litecho — reconstructed local audio converter

Flask app with an accessible HTML interface, TXT/PDF extraction, optional image OCR, and MP3 generation through gTTS.

```sh
python3 -m pip install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:5001`. Paste text or extract it from a file, then create audio. Image OCR additionally requires the Tesseract system executable. Scanned PDF OCR is not implemented. File uploads are limited to 10 MiB; text is limited to 10000 characters. No files are retained by the application.

`POST /api/extract`: multipart field `file`. `POST /api/audio`: JSON `{"text":"Hello"}` returning MP3 bytes. gTTS needs internet access and sends the text to Google's speech service; avoid private text unless that transmission is acceptable to you.

This reconstruction replaces the remembered React UI with a small HTML interface. It does not claim to restore the original Firebase/Render deployment. Run locally; there is no authentication or public-service hardening.
