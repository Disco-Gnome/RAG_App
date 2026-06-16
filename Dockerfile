# Use a Docker-official Python image
FROM python:3.10-bookworm

COPY data/ ./data

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /src/
# before we copy the rest of the project for optimization
COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install project dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY .env .
COPY src/ ./


# Make port 8501 available to the world outside this container because that is the streamlit default port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Make the program executable, as per Streamlit docs
# https://docs.streamlit.io/deploy/tutorials/docker
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
