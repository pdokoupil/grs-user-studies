from flask import Flask
import os
from flask_pluginkit import PluginManager

app = Flask(__name__)
pm = PluginManager(plugins_folder="src/plugins")
pm.init_app(app)


@app.route("/limit2")
def lim():
    return "Hello from the main app"

@app.route("/")
def index():
    print("XY")
    return "DEF"

if __name__ == "__main__":
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    print(os.listdir("src/plugins"))
    print(f"Starting, all plugins: {pm.get_all_plugins}")
    app.run(debug=True, host="0.0.0.0")