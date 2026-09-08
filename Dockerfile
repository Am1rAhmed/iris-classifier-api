FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Using 0.0.0.0 instead of 127.0.0.1: inside a container, 127.0.0.1 (localhost)
# only refers to the container's own internal loopback interface. It would
# make the app reachable only from *within* the container itself, not from
# the host machine or any port mapping. 0.0.0.0 tells uvicorn to listen on
# ALL network interfaces inside the container, so Docker's port mapping
# (-p 8000:8000) can actually forward host traffic into the app.
