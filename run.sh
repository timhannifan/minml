#!/bin/bash

virtualenv env
source env/bin/activate

pip install -r requirements.txt

cd src/minml
python3 main.py --config "../../examples/donors/config.yaml" --db "../../examples/donors/config_db.yaml" --load_db 1

deactivate
