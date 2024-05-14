import pathlib
import tomllib

path = pathlib.Path(__file__).parent / "parameters.toml"

with path.open(mode = "rb") as f:
    parameters = tomllib.load(f)
