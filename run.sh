#!/bin/bash

virtualenv env
source env/bin/activate

pip install -r requirements.txt


cd src/minml
python3 main.py --config "../../examples/donors/config.yaml" --in_path "../../examples/donors/data.csv" --out_path '../../examples/out' --test 1

deactivate
