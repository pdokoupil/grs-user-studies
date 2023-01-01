# -*- coding: utf-8 -*-

import os
import random
import re
from flask import Blueprint, jsonify, request, redirect, current_app, url_for, make_response, render_template
from flask_login import current_user
from itsdangerous import URLSafeTimedSerializer
import numpy as np
import requests
from app import csrf
from common import gen_url_prefix, load_languages, multi_lang
from flask_wtf.csrf import generate_csrf
import flask_wtf.csrf
import flask
import json

import datetime

from models import Interaction, Participation
from app import db

from .popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation



__plugin_name__ = "utils"
__description__ = "Plugin containing common, shared functionality that can be used from other plugins."
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

from .preference_elicitation import load_data_1, load_data_2, load_data_3, recommend_1, recommend_2_3, search_for_movie, rlprop, weighted_average, calculate_weight_estimate, result_layout_variants


NUM_TO_SELECT = 5

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

languages = load_languages(os.path.dirname(__file__))
print(f"Languages={languages}")

def get_lang():
    default_lang = "en"
    if "lang" in flask.session and flask.session["lang"] and flask.session["lang"] in languages:
        return flask.session["lang"]
    return default_lang


# Returns translator function for translating phrases to given language
def get_tr(lang):
    def tr(phrase):
        return languages[lang][phrase]
    return tr

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
    

    tr = get_tr(get_lang())
    params["title"] = tr("join_title")
    params["participant_details"] = tr("join_participant_details")
    params["please_enter_details"] = tr("join_please_enter_details")
    params["enter_email"] = tr("join_enter_email")
    params["enter_email_hint"] = tr("join_enter_email_hint")
    params["enter_gender"] = tr("join_enter_gender")
    params["enter_gender_hint"] = tr("join_enter_gender_hint")
    params["enter_age"] = tr("join_enter_age")
    params["enter_age_hint"] = tr("join_enter_age_hint")
    params["enter_education"] = tr("join_enter_education")
    params["enter_education_hint"] = tr("join_enter_education_hint")
    params["enter_ml_familiar"] = tr("join_enter_ml_familiar")
    params["enter_ml_familiar_hint"] = tr("join_enter_ml_familiar_hint")
    params["gender_male"] = tr("join_gender_male")
    params["gender_female"] = tr("join_gender_female")
    params["gender_other"] = tr("join_gender_other")
    params["education_no_formal"] = tr("join_education_no_formal")
    params["education_primary"] = tr("join_education_primary")
    params["education_high"] = tr("join_education_high")
    params["education_bachelor"] = tr("join_education_bachelor")
    params["education_master"] = tr("join_education_master")
    params["education_doctoral"] = tr("join_education_doctoral")
    params["yes"] = tr("yes")
    params["no"] = tr("no")
    params["informed_consent_header"] = tr("join_informed_consent_header")
    params["informed_consent_p1"] = tr("join_informed_consent_p1")
    params["informed_consent_p2"] = tr("join_informed_consent_p2")
    params["informed_consent_p3"] = tr("join_informed_consent_p3")
    params["informed_consent_p31"] = tr("join_informed_consent_p31")
    params["informed_consent_p32"] = tr("join_informed_consent_p32")
    params["informed_consent_p33"] = tr("join_informed_consent_p33")
    params["informed_consent_p4"] = tr("join_informed_consent_p4")
    params["informed_consent_p5"] = tr("join_informed_consent_p5")
    params["informed_consent_p6"] = tr("join_informed_consent_p6")
    params["start_user_study"] = tr("join_start_user_study")

    print(f"Final params={params}")
    
    

    return render_template("join.html", **params)

@bp.route("/preference-elicitation", methods=["GET", "POST"])
@multi_lang # TODO remove? and keep only in plugin1
def preference_elicitation():
    json_data =  {} #request.get_json()
    impl = request.args.get("impl") or 1
    flask.session["elicitation_movies"] = []
    
    params = {
        "impl": impl,
        "consuming_plugin": "plugin1"
    }
    
    tr = get_tr(get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliary_university"] = tr("footer_cagliary_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["load_more"] = tr("elicitation_load_more")
    params["finish"] = tr("elicitation_finish")
    params["search"] = tr("elicitation_search")
    params["cancel_search"] = tr("elicitation_cancel_search")
    params["enter_name"] = tr("elicitation_enter_name")
    params["header"] = tr("elicitation_header")
    params["hint"] = tr("elicitation_hint")

    return render_template("preference_elicitation.html", **params) # TODO remove hardcoded consuming plugin

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


@bp.route("/changed-viewport", methods=["POST"])
def changed_viewport():
    print("## Called viewport changed")
    print(f"Passed data= {request.get_json()}")

    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "changed-viewport", #InteractionType.query.filter(InteractionType.name == "changed-viewport").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json())
    )
    db.session.add(x)
    db.session.commit()

    return "OK"

@bp.route("/selected-item", methods=["POST"])
def selected_item():
    print(f"GOT={request.get_json()}")
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "selected-item", #InteractionType.query.filter(InteractionType.name == "selected-item").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json())
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

@bp.route("/deselected-item", methods=["POST"])
def deselected_item():
    print(f"GOT={request.get_json()}")
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "deselected-item", #InteractionType.query.filter(InteractionType.name == "deselected-item").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json())
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

@bp.route("/rl-prop", methods=["GET"])
def rl_prop():
    x = rlprop()
    return x

@bp.route("/loaded-page", methods=["POST"])
def loaded_page():
    print(f"GOT={request.get_json()}")
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "loaded-page", #InteractionType.query.filter(InteractionType.name == "loaded-page").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json())
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

@bp.route("/on-input", methods=["POST"])
def on_input():
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "on-input",
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json())
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

# Receives arbitrary feedback (typically from preference elicitation) and generates recommendation
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

    # Movie indices of selected movies
    selected_movies = request.args.get("selectedMovies")
    selected_movies = selected_movies.split(",") if selected_movies else []
    selected_movies = [int(m) for m in selected_movies]

    # Calculate weights based on selection and shown movies during preference elicitation
    weights = calculate_weight_estimate(selected_movies, flask.session["elicitation_movies"])
    flask.session["weights"] = weights.tolist()
    print(f"### Weights are estimated to {weights}")

    # recommended_items = recommend_2_3(selected_movies)

    # #return recommended_items
    # print(f"Recommended items: {recommended_items}")

    algorithms = ["relevance_based", "rlprop", "weighted_average"]
    # Add default entries so that even the non-chosen algorithm has an empty entry
    # to unify later access
    recommendations = {
        algo: [[]] for algo in algorithms
    }
    
    # We always take relevance_based algorithm and add one randomly chosen algorithm to it
    rnd_algorithms = algorithms[1:] # Randomly choosing between rlprop and weighted_average
    random.shuffle(rnd_algorithms)
    algorithms = ["rlprop", "weighted_average"] #algorithms[:1] + rnd_algorithms[:1] # Take relevance_based + one random algorithm
    print(f"Chosen algorithms = {algorithms}")
    
    # We filter out everything the user has selected during preference elicitation.
    # However, we allow future recommendation of SHOWN, NOT SELECTED (during elicitation, not comparison) altough these are quite rare
    filter_out_movies = selected_movies

    # Order of insertion should be preserved
    recommended_items, model = recommend_2_3(selected_movies, filter_out_movies, return_model=True)
    for algorithm in algorithms:
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(selected_movies, model, weights, filter_out_movies)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(selected_movies, model, weights, filter_out_movies)
        else:
            assert False
        recommendations[algorithm] = [recommended_items]

    
    flask.session["movies"] = recommendations
    #flask.session["movies"] = [recommended_items]
    
    
    flask.session["iteration"] = 1
    # TODO store all these information in DB as well
    flask.session["elicitation_selected_movies"] = selected_movies
    print("### zeroing")
    flask.session["selected_movie_indices"] = [] #dict() # For each iteration, we can store selected movies

    flask.session["permutation"] = np.random.permutation(len(result_layout_variants)).tolist()

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