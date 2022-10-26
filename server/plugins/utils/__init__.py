# -*- coding: utf-8 -*-

import os
import random
import re
from flask import Blueprint, jsonify, request, redirect, current_app, url_for, make_response, render_template
from flask_login import current_user
from itsdangerous import URLSafeTimedSerializer
import requests
from app import csrf
from common import gen_url_prefix
from flask_wtf.csrf import generate_csrf
import flask_wtf.csrf
import flask
import json

__plugin_name__ = "utils"
__description__ = "Plugin containing common, shared functionality that can be used from other plugins."
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

from .preference_elicitation import load_data


NUM_TO_SELECT = 5

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

# Shared implementation of "/join" phase of the user study
# Expected input is continuation_url
# Expected output is 
@bp.route("/join", methods=["GET"])
def join():
    assert "continuation_url" in request.args, f"Continuation url must be available: {request.args}"
    assert "guid" in request.args, f"Guid must be available: {request.args}"

    params = dict(request.args)
    print(f"Params={params}")
    params["email"] = current_user.email if current_user.is_authenticated else ""
    # if current_user.is_authenticated:
    #     params["email"] = current_user.email
        # with current_app.test_client() as c:
        #     rv = c.get("/login")
        #     m = re.search(b'<input type="hidden" name="csrf_token" value="(.*)" />', rv.data)
        #     csrf_token = m.group(1).decode('utf-8')
            
        #     json_data = {"some_kez": 123}
            
        #     resp = c.post(
        #         "/add-participant",
        #         json=json_data, follow_redirects=True, headers={'X-CSRFToken': csrf_token})
        #     print(f"Resp={resp}")
    print(f"Final params={params}")
    return render_template("join.html", **params)

@bp.route("/preference-elicitation", methods=["GET"])
def preference_elicitation():
    json_data =  {} #request.get_json()
    return render_template("preference_elicitation.html")

@bp.route("/cluster-data", methods=["GET"])
def cluster_data():
    return json.dumps(load_data())
    
    #return render_template("preference_elicitation.html", **json_data)

# Shared implementation of "/finish" phase of the user study
@bp.route("/finish", methods=["GET", "POST"])
def finish():
    json_data = request.get_json()
    return render_template("finish.html", **json_data)
    

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