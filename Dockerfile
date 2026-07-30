FROM gcr.io/deeplearning-platform-release/tf-cpu.2-15.py310
WORKDIR /
ENV GCLOUD_PROJECT=banca-march-379915

# Install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy shared utilities
COPY model_metadata.py /model_metadata.py

# Copy trainer code
COPY trainer /trainer

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
