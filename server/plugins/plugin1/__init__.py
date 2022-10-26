# -*- coding: utf-8 -*-

import os
from flask import Blueprint, jsonify, request, redirect, url_for, make_response, render_template


__plugin_name__ = "plugin1"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

NUM_TO_SELECT = 5

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

@bp.route("/create")
def create():
    return render_template("create.html")

@bp.route("/num-to-select")
def get_num_to_select():
    return {
        'num_to_select': NUM_TO_SELECT
    }

@bp.route("/limit")
def limit():
    template_vars = {
        "title": __plugin_name__,
        "author": __author__,
        "version": __version__,
        "author_contact": __author_contact__,
        "num_to_select": NUM_TO_SELECT
    }
    return render_template("plugin1.html", **template_vars)
    #return jsonify(dict(status=0, message="Access Denial"))

@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    guid = request.args.get("guid")
    return redirect(url_for("utils.join", continuation_url=url_for("plugin1.preference_elicitation"), guid=guid))

@bp.route("/step1", methods=["GET", "POST"])
def step1():
    # Parameters received from callback (if this is continuation) are in JSON
    json_data = request.get_json()
    print(f"Got json: {json_data}")
    print("After preference elicitation")
    return render_template("step.html", step_number=1)

@bp.route("/step2", methods=["GET"])
def step2():
    return render_template("step.html", step_number=2)

@bp.route("/preference-elicitation", methods=["GET", "POST"])
def preference_elicitation():
    # Redirect preference elicitation to common implementation from utils
    json_data = request.get_json()
    assert "email" in json_data, f"Email must be available in request {json_data}"
    assert "guid" in json_data, f"Guid must be available in request {json_data}"

    guid = json_data["guid"]
    print(f"In plugin1.preference_elicitation, got json_data={json_data}")
    return redirect(url_for("utils.preference_elicitation", continuation_url=url_for("plugin1.step1"), guid=guid, email=json_data["email"]))

def limit_handler():
    """I am running in before_request"""
    ip = request.headers.get('X-Real-Ip', request.remote_addr)
    if request.endpoint == "index" and ip == "127.0.0.1":
        resp = make_response(redirect(url_for("plugin1.limit")))
        resp.is_return = True
        return resp

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
        "hep": dict(before_request=limit_handler)
    }