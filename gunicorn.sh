#!/bin/sh
source bin/activate
nohup gunicorn --chdir app wsgi:app -w 2 --threads 2 -b 0.0.0.0:8000 > brand-classification.log 2>&1&

