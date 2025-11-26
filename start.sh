#!/usr/bin/env bash

# Install gunicorn for production management
pip install gunicorn

# Run Uvicorn using Gunicorn with 4 workers for high concurrency and stability.
# --bind 0.0.0.0:$PORT ensures the app listens on the port assigned by the host (e.g., Render).
# main:app should now be replaced with app:app since your file is named app.py.
exec gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
