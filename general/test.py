import pathlib

f = pathlib.Path('test_configs.json')
print(f.read_text())
