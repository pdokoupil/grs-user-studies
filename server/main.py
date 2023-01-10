import bcrypt
import flask
import os
from flask_pluginkit import PluginManager

import yaml

from flask_login import current_user, login_required

import secrets

import sqlalchemy

from app import pm, db, csrf
from models import Participation, UserStudy
from common import gen_url_prefix

import json

import datetime

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
        return flask.redirect(flask.url_for('auth.login'))


def get_loaded_plugins():
    endpoints = {str(p) for p in flask.current_app.url_map.iter_rules()}
    return [{
        "plugin_name": p["plugin_name"],
        "plugin_description": p["plugin_description"],
        "plugin_version": p["plugin_version"],
        "plugin_author": p["plugin_author"],
        "create_url": f"/{p['plugin_name']}/create"
    } for p in pm.get_enabled_plugins if f"/{p['plugin_name']}/create" in endpoints]


def get_loaded_plugin_names():
    return {p["plugin_name"] for p in get_loaded_plugins()}

# Returns a list of dicts (JSON) containing information about loaded plugins
# Only enabled plugins that also has pluginName/create endpoint defined, are listed
@main.route("/loaded-plugins")
def loaded_plugins():
    return get_loaded_plugins()


# Returns ids of all existing (both current and past) user studies
@main.route("/existing-user-studies")
@login_required
def existing_user_studies():
    result = db.session.query(UserStudy, sqlalchemy.func.count(Participation.participant_email)).outerjoin(Participation, UserStudy.id==Participation.user_study_id).group_by(UserStudy.id).all()
    # result_json = []
    # for user_study, num_participants in result:
    #     result_json.append({

    #     })
    # return result_json
    
    # Filtering condition
    # Admin users see all the user studies while
    # normal users see only user studies created by them
    def filter_cond(x):
        if current_user.is_admin():
            return True
        return x.creator == current_user.get_id()

    return flask.jsonify([{
            "id": x.id,
            "creator": x.creator,
            "guid": x.guid,
            "parent_plugin": x.parent_plugin,
            "settings": x.settings,
            "time_created": x.time_created,
            "participants": c,
            "join_url": gen_user_study_invitation_url(x.parent_plugin, x.guid)
        } for x, c in result if filter_cond(x)])

def gen_user_study_url(guid):
    return f"/user-study/{guid}"

def gen_user_study_invitation_url(parent_plugin, guid):
    return f"{gen_url_prefix()}/{parent_plugin}/join?guid={guid}"

def get_vars(x):
    return {name: value for name, value in vars(x).items() if not name.startswith("_")}

@main.route("/user-study", methods=["GET"])
def get_user_study():
    user_study_id = flask.request.args.get("user_study_id")
    studies = UserStudy.query.filter(UserStudy.id == user_study_id).all()
    assert len(studies) <= 1
    if studies:
        return flask.jsonify(get_vars(studies[0]))
    else:
        return "Not found", 404

@main.route("/user-studies", methods=["GET"])
def get_user_studies():
    studies = UserStudy.query.all()
    return flask.jsonify([get_vars(x) for x in studies])

@main.route("/participations", methods=["GET"])
def get_participations():
    participations = Participation.query.all()
    return flask.jsonify([get_vars(x) for x in participations])

@main.route("/user-study-participants", methods=["GET"])
def get_user_study_participants():
    user_study_id = flask.request.args.get("user_study_id")
    participants = Participation.query.filter(Participation.user_study_id == user_study_id).with_entities(Participation.participant_email).all()
    return flask.jsonify([{"participant_email": x[0]} for x in participants])

# Returns user studies in which the given user participated
@main.route("/user-participated-user-studies", methods=["GET"])
def get_user_participated_user_studies():
    user_email = flask.request.args.get("user_email")
    studies = Participation.query.filter(Participation.participant_email == user_email).with_entities(Participation.user_study_id)
    return flask.jsonify([{"user_study_id": x[0]} for x in studies])

# Adds a record that user starts participation in a user study
@main.route("/add-participant", methods=["POST"])
def add_participant():
    json_data = flask.request.get_json()
    print(f"Json data is: {json_data}")
    
    user_study = UserStudy.query.filter(UserStudy.guid == json_data["user_study_guid"]).first()

    if not user_study:
        return "GUID not found", 404
    
    user_study_id = user_study.id

    print(f"from guid={json_data['user_study_guid']} we got id={user_study_id}")

    participation = Participation(
        participant_email=json_data["user_email"],
        user_study_id=user_study_id,
        time_joined=datetime.datetime.utcnow(),
        time_finished=None,
        age_group=json_data["age_group"],
        gender=json_data["gender"],
        education=json_data["education"],
        ml_familiar=json_data["ml_familiar"],
        language=json_data["lang"],
        uuid=flask.session["uuid"]
    )
    print("Participation created")
    db.session.add(participation)
    db.session.commit()
    
    flask.session["participation_id"] = participation.id

    return "OK"

# Global create handler - takes user study settings and creates an user study from it
# Usually called from the individual plugins' create handlers
@main.route("/create-user-study", methods=["POST"])
@login_required
def create_user_study():
    guid = secrets.token_urlsafe(24)
    json_data = flask.request.get_json()

    if "parent_plugin" not in json_data:
        return "Bad Request - parent plugin was not specified", 400

    if json_data["parent_plugin"] not in get_loaded_plugin_names():
        return "Bad Request - invalid parent plugin", 400

    if "config" not in json_data:
        # No config was specified for the user study
        json_data["config"] = dict()

    study = UserStudy(
        creator=current_user.email,
        guid=guid,
        parent_plugin=json_data["parent_plugin"],
        settings = json.dumps(json_data["config"]),
        time_created = datetime.datetime.utcnow()
    )
    
    db.session.add(study)
    db.session.commit()

    # return {
    #     "status": "success",
    #     "url": gen_user_study_url(guid)
    # }

    return flask.redirect(flask.url_for('main.administration'), Response={
        "status": "success",
        "url": gen_user_study_url(guid)
    })

    

# if __name__ == "__main__":
#     print(os.getcwd())
#     print(os.listdir(os.getcwd()))
#     print(os.listdir("plugins"))
#     print(f"Starting, all plugins: {pm.get_all_plugins}")
#     app.run(debug=True, host="0.0.0.0")