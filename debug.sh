./src/gputls/move.sh
python3 -m pip uninstall -y gputls
python3 -m build
python3 -m pip install dist/gputls-0.2.0-py3-none-any.whl
# python3 test1.py