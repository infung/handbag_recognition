# Setup

Python 3.11 - 12 (Debian)

```bash
sudo apt update && sudo apt upgrade
sudo apt install python3
sudo apt install python3-pip
./setup.sh
```

# How to run

```bash
./gunicorn.sh
```

# How to test

```bash
curl -i -X POST -H "Content-Type: multipart/form-data"  -F "file=@{path-to-image}" localhost:5000/api/predict
```

# Utility

```bash
# to copy file from GCP bucket
sudo gsutil cp gs://{buceket-name}/{folder}}/{file-name} {path-in-local}

# to get pid of gunicorn server
ps aux | grep gunicorn

# to kill process by pid
sudo kill -9 {pid}
```
