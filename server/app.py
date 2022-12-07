import flask
from flask_pluginkit import PluginManager
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from flask_session import Session

db = SQLAlchemy()
pm = PluginManager(plugins_folder="plugins")
csrf = CSRFProtect()

sess = Session()

def create_app():
    app = flask.Flask(__name__)

    app.config['SECRET_KEY'] = '8bf29bd88d0bfb94509f5fb0'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    app.config['SESSION_COOKIE_NAME'] = "something"
    app.config["SESSION_TYPE"] = "filesystem"

    sess.init_app(app)

    db.init_app(app)

    csrf.init_app(app)

    login_manager = LoginManager(app)
    
    pm.init_app(app)


    from models import User

    @login_manager.user_loader
    def user_loader(user_id):
        """Given *user_id*, return the associated User object.

        :param unicode user_id: user_id (email) user to retrieve

        """
        return User.query.get(user_id)

    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    with app.app_context():
        db.create_all()

    return app