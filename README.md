# minml
Barebones machine learning pipeline


## Usage
```
git clone https://github.com/timhannifan/minml
cd minml
sh run.sh

<Or for ipython>
virtualenv env
source env/bin/activate
pip install -r requirements.txt
cd src/minml

ipython3
run main --config "../../examples/donors/config.yaml" --db "../../examples/donors/db_config.yaml" --load_db 1
```
