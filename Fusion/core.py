import importlib.util
import os

_here = os.path.dirname(__file__)
_pyd_path = os.path.join(_here, "Fusion.pyd")

if not os.path.exists(_pyd_path):
    raise ImportError("Fusion.pyd not found in Fusion directory.")

spec = importlib.util.spec_from_file_location("Fusion", _pyd_path)
Fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Fusion)