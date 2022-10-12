import bcrypt
import flask
import os
from flask_pluginkit import PluginManager






main = flask.Blueprint('main', __name__)

@main.route("/limit2")
def lim():
    return flask.render_template("index.html")

@main.route("/")
def index():
    print("XY")
    return flask.render_template("index.html")

# @app.route("/administration")
# @login_required
# def administration():
#     return render_template("administration.html")

@main.route("/administration")
def administration():
    if current_user.is_authenticated:
        return flask.render_template("administration.html")
    else:
        return flask.render_template("login.html")


# if __name__ == "__main__":
#     print(os.getcwd())
#     print(os.listdir(os.getcwd()))
#     print(os.listdir("plugins"))
#     print(f"Starting, all plugins: {pm.get_all_plugins}")
#     app.run(debug=True, host="0.0.0.0")