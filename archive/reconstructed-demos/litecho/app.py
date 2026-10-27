"""Reconstructed accessible text/PDF/image-to-audio local application."""
import io
from flask import Flask, jsonify, render_template, request, send_file


def extract_text(upload):
    suffix=upload.filename.lower().rsplit('.',1)[-1]
    if suffix=='txt':return upload.read().decode('utf-8-sig')
    if suffix=='pdf':
        from pypdf import PdfReader
        return '\n'.join(page.extract_text() or '' for page in PdfReader(upload.stream).pages)
    if suffix in {'png','jpg','jpeg'}:
        from PIL import Image
        import pytesseract
        return pytesseract.image_to_string(Image.open(upload.stream))
    raise ValueError('Supported file types: TXT, PDF, PNG, JPG')


def create_app(synthesize=None):
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=10*1024*1024
    def default_synthesize(text):
        from gtts import gTTS
        buffer=io.BytesIO();gTTS(text=text,lang='en').write_to_fp(buffer)
        return buffer.getvalue()
    audio=synthesize or default_synthesize

    @app.get('/')
    def home():return render_template('index.html')

    @app.post('/api/extract')
    def extract():
        upload=request.files.get('file')
        if upload is None or not upload.filename:return jsonify(error='Choose a file'),400
        try:
            text=extract_text(upload).strip()
            if not text:return jsonify(error='No text found; scanned PDFs require OCR first'),400
            return jsonify(text=text[:10000],truncated=len(text)>10000)
        except Exception as exc:
            app.logger.info('Extraction failed: %s',type(exc).__name__)
            return jsonify(error='Unable to extract this file. Check its format and OCR setup.'),400

    @app.post('/api/audio')
    def convert():
        data=request.get_json(silent=True) or {}
        text=data.get('text','')
        if not isinstance(text,str) or not 1<=len(text.strip())<=10000:
            return jsonify(error='Provide 1–10000 characters'),400
        try:return send_file(io.BytesIO(audio(text.strip())),mimetype='audio/mpeg',as_attachment=True,download_name='litecho.mp3')
        except Exception:
            return jsonify(error='Speech service unavailable. Try again later.'),502
    return app


if __name__=='__main__':create_app().run(host='127.0.0.1',port=5001,debug=False)
