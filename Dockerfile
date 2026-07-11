# Hugging Face Spaces (Docker SDK) entrypoint. Runs the same serve.py the
# Render deploy uses. HF sets the app port to 7860 (see README app_port).
#
# Follows HF's recommended non-root user pattern so HOME and the repo cache are
# writable at runtime (Streamlit writes to ~/.streamlit; the app reads the
# committed cache/ and never needs /data here).
FROM python:3.12.8-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "serve.py"]
