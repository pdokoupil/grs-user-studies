import glob
import json
import os
import yaml 
import functools

from flask import request, session

def load_config(file_name="config.yaml"):
    with open(file_name, "r") as f:
        return yaml.safe_load(f)

yaml_config = load_config()
print(f"Yaml config={yaml_config}")

def gen_url_prefix():
    return f"http://{yaml_config['hostname']}:{yaml_config['port']}"

def load_languages(base_path):
    res = {}
    for x in glob.glob(os.path.join(base_path, "static/languages/*.json")):
        with open(x, "r", encoding="utf8") as f:
            res[os.path.splitext(os.path.basename(x))[0]] = json.loads(f.read())
    return res

def multi_lang(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        lang = request.args.get('lang')
        print(f"### Language = '{lang}'")
        if lang:
            session["lang"] = lang
        return func(*args, **kwargs)
    return inner