git submodule update --init
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
./build.sh
python3 app/gui.py