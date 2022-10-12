import flask
from flask_pluginkit import PluginManager
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()

def create_app():
    app = flask.Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    db.init_app(app)

    login_manager = LoginManager(app)
    
    print("@@ Called")
    pm = PluginManager(plugins_folder="plugins")
    pm.init_app(app)
    print("@@ Still called")

    from models import User

    @login_manager.user_loader
    def user_loader(user_id):
        """Given *user_id*, return the associated User object.

        :param unicode user_id: user_id (email) user to retrieve

        """
        return User.query.get(user_id)

    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app