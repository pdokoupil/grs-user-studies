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

from .popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation

__plugin_name__ = "utils"
__description__ = "Plugin containing common, shared functionality that can be used from other plugins."
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

from .preference_elicitation import load_data_1, load_data_2, load_data_3, recommend_1, recommend_2_3, search_for_movie


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
    impl = request.args.get("impl") or 1
    flask.session["elicitation_movies"] = []
    print("@@ Called")
    return render_template("preference_elicitation.html", impl=impl)

@bp.route("/cluster-data-1", methods=["GET"])
def cluster_data_1():
    #return json.dumps(load_data_1())
    el_movies = flask.session["elicitation_movies"]
    
    x = load_data_1(el_movies)
    el_movies.extend(x)
    flask.session["elicitation_movies"] = el_movies

    # TODO to do lazy loading, return just X and update rows & items in JS directly
    return jsonify(el_movies)

    #return render_template("preference_elicitation.html", **json_data)

@bp.route("/cluster-data-2", methods=["GET"])
def cluster_data_2():
    el_movies = flask.session["elicitation_movies"]
    
    x = load_data_2(el_movies)
    el_movies.extend(x)
    flask.session["elicitation_movies"] = el_movies
    return jsonify(el_movies)

@bp.route("/cluster-data-3", methods=["GET"])
def cluster_data_3():
    el_movies = flask.session["elicitation_movies"]
    
    x = load_data_3(el_movies)
    el_movies.extend(x)
    flask.session["elicitation_movies"] = el_movies

    return jsonify(el_movies)

@bp.route("/send-feedback", methods=["GET"])
def send_feedback():
    print(f"Args={request.args}")
    impl = request.args.get("impl")
    print(f"Impl='{impl}'")
    # if impl == "1":
    #     print(f"IMPL = 1")
    #     selected_cluster = int(request.args.get("selectedCluster"))
    #     recommended_items = recommend_1(selected_cluster)
    # elif impl == "2" or impl == "3":
    #     print("IMPL = 2/3")
    #     selected_movies = request.args.get("selectedMovies").split(",")
    #     selected_movies = [int(m) for m in selected_movies]
    #     recommended_items = recommend_2_3(selected_movies)
    selected_movies = request.args.get("selectedMovies").split(",")
    selected_movies = [int(m) for m in selected_movies]
    recommended_items = recommend_2_3(selected_movies)

    #return recommended_items
    print(f"Recommended items: {recommended_items}")
    flask.session["movies"] = [recommended_items]
    flask.session["iteration"] = 1
    # TODO store all these information in DB as well
    flask.session["elicitation_selected_movies"] = selected_movies
    print("### zeroing")
    flask.session["selected_movie_indices"] = [] #dict() # For each iteration, we can store selected movies
    return redirect(url_for("plugin1.compare_algorithms"))
    #return redirect(url_for("plugin1.compare_algorithms", movies=recommended_items))

# Shared implementation of "/finish" phase of the user study
@bp.route("/finish", methods=["GET", "POST"])
def finish():
    json_data = request.get_json()
    return render_template("finish.html", **json_data)
    
@bp.route("/movie-search", methods=["GET"])
def movie_search():
    attrib = flask.request.args.get("attrib")
    pattern = flask.request.args.get("pattern")
    if not attrib or attrib not in ["movie"]: # TODO extend search support
        return make_response("", 404)
    if not pattern:
        return make_response("", 404)
    
    res = search_for_movie(attrib, pattern)

    return flask.jsonify(res)

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