import pathlib
import json
import sys

class Configs(dict):
    def __init__(self, init_dict=None, *args, **kwargs):
        if init_dict is not None:
            self._dict = {}
            for k, v in init_dict.items():
                self._dict[k] = v
        else:
            self._dict = dict(**kwargs)
    
    def save(self, filename):
        f = pathlib.Path(filename)
        if f.suffix == '.json':
            f.write_text(json.dumps(dict(self)))
        elif f.suffix in ['.yml', '.yaml']:
            import ruamel.yaml as yaml
            with f.open('w') as f_:
                yaml.safe_dump(dict(self), f_)
        else:
            raise NotImplementedError(f.suffix)

    @classmethod
    def load_from_file(cls, filename):
        f = pathlib.Path(filename)
        if f.suffix == '.json':
            return cls(json.loads(f.read_text()))
        elif f.suffix in ['.yml', '.yaml']:
            import ruamel.yaml as yaml
            return cls(yaml.safe_load(f.read_text()))
        else:
            raise NotImplementedError(f.suffix)
    
    @classmethod
    def load(cls, preset_file_path=None, argv=None):
        if argv is None:
            argv = sys.argv
        argv = argv[1:]

        if len(argv) >= 1 and (argv[0] == 'config_preset' or argv[0] == '--config_preset'):
            preset_file_path = argv[1]
            argv = argv[2:]
        if preset_file_path is not None:
            config = cls.load_from_file(preset_file_path)
        else:
            config = cls()
        
        key = None
        vals = []
        for arg in argv:
            if arg.startswith('--'):
                arg = arg[2:]
                tokens = arg.split('=')
                if len(tokens) >= 2:
                    config.update_key_values(tokens[0], [config.parse_token(tokens[1])])
                    key = None
                    continue

                if key:
                    config.update_key_values(key, vals)
                key, vals = arg, []
            else:
                assert key, 'argument value is given without key'
                vals.append(config.parse_token(arg))
        if key:
            config.update_key_values(key, vals)
        return config

    def update_key_values(self, key, values):
        if len(values) == 0:
            self[key] = True
        elif len(values) == 1:
            self[key] = values[0]
        else:
            self[key] = values
    
    def merge(self, other_configs):
        for k, v in other_configs.items():
            self[k] = v


    def parse_token(self, token):
        # bool test
        if token.lower() == 'false':
            return False
        if token.lower() == 'true':
            return True
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            pass
        return token
    
    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def items(self):
        return self._dict.items()

    def to_json(self):
        return json.dumps(self._dict, sort_keys=True, indent=4)

    def __getitem__(self, key):
        if key in self._dict:
            return self._dict[key]
        else:
            return None

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)
        message = f"Tried to set key '{key}' on immutable config. Use update()."
        raise AttributeError(message)

    def __str__(self):
        return self._dict.__str__()


