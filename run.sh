#!/bin/bash

virtualenv env
source env/bin/activate

pip install -r requirements.txt

cd src/minml
python3 main.py --config "../../examples/donors/config/config_test_env.yaml" --db "../../examples/donors/config/config_db.yaml"

deactivate
