import yaml 
import functools

from flask import request

def load_config(file_name="config.yaml"):
    with open(file_name, "r") as f:
        return yaml.safe_load(f)

yaml_config = load_config()
print(f"Yaml config={yaml_config}")

def gen_url_prefix():
    return f"http://{yaml_config['hostname']}:{yaml_config['port']}"


def multi_lang(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        print(f"### Language = '{request.args.get('lang')}'")
        return func(*args, **kwargs)
    return inner