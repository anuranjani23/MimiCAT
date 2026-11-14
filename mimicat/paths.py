import os

# Folder that contains one sub-folder per character: <root>/<name>/<name>.npz and <name>/motion/*.npz
DATA_ROOT = os.environ.get("MimiCAT", "data/PokeAnimDB/data")
