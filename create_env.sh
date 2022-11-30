#!/bin/bash

python3 -m venv faust-partial-env
source faust-partial-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt