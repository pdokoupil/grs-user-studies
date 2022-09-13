from flask import Flask, render_template
import os
from flask_pluginkit import PluginManager

app = Flask(__name__)
pm = PluginManager(plugins_folder="plugins")
pm.init_app(app)


@app.route("/limit2")
def lim():
    return render_template("index.html")

@app.route("/")
def index():
    print("XY")
    return render_template("index.html")

if __name__ == "__main__":
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    print(os.listdir("plugins"))
    print(f"Starting, all plugins: {pm.get_all_plugins}")
    app.run(debug=True, host="0.0.0.0")