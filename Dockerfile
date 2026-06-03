# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install torch (CPU only) first to save hundreds of megabytes
RUN pip install --default-timeout=1000 --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container (for Streamlit)
ENV PYTHONPATH=/app

# Define environment variable
ENV NAME World

# Run streamlit_app.py when the container launches
CMD ["streamlit", "run", "economic_graphrag/ui/streamlit_app.py"]
