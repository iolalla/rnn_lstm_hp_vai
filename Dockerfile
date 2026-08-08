FROM gcr.io/deeplearning-platform-release/tf-cpu.2-15.py310
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared utilities
COPY model_metadata.py /app/model_metadata.py

# Copy trainer package
COPY trainer /app/trainer

# Default entry point
ENTRYPOINT ["python", "-m", "trainer.task"]

