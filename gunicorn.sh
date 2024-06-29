#!/bin/sh

ps aux | grep gunicorn | awk '{print $2}' | xargs sudo kill -9

source ./bin/activate

nohup gunicorn --chdir app wsgi:app -w 2 --threads 2 -b 0.0.0.0:5000 > brand-classification.log 2>&1&
