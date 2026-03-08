# Use Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy your requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your app.py and your model weights
COPY . .

# Start the FastAPI server on port 7860 (Hugging Face's default port)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
