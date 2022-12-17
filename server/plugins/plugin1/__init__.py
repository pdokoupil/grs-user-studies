# -*- coding: utf-8 -*-

import datetime
import json
import os
import time
from flask import Blueprint, jsonify, request, redirect, url_for, make_response, render_template, session

from plugins.utils.preference_elicitation import recommend_2_3

from models import Interaction, Participation
from app import db, pm
from common import multi_lang
import glob


__plugin_name__ = "plugin1"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

NUM_TO_SELECT = 5

MIN_ITERATION_TO_CANCEL = 5
TOTAL_ITERATIONS = 8

HIDE_LAST_K = 1000000 # Effectively hides everything

def load_languages():
    print("PARSING LANGUAGES #########")
    res = {}
    for x in glob.glob(os.path.join(os.path.dirname(__file__), "static/languages/*.json")):
        with open(x, "r", encoding="utf8") as f:
            res[os.path.splitext(os.path.basename(x))[0]] = json.loads(f.read())
    return res

languages = load_languages()
print(f"Languages={languages}")

@bp.before_app_first_request
def bp_init():
    print(pm.emit_assets("plugin1", "languages/en.json"))

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
@multi_lang
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    guid = request.args.get("guid")
    return redirect(url_for("utils.join", continuation_url=url_for("utils.preference_elicitation"), guid=guid))

@bp.route("/step1", methods=["GET", "POST"])
def step1():
    # Parameters received from callback (if this is continuation) are in JSON
    # json_data = request.get_json()
    # print(f"Got json: {json_data}")
    # print("After preference elicitation")
    return render_template("step.html", step_number=1, movies=session["movies"][-1])

@bp.route("/compare-algorithms", methods=["GET"])
def compare_algorithms():
    movies = [session["movies"][-1]]
    movies.append([movies[0][0]] * len(movies[0]))
    result_layout = request.args.get("result_layout")
    result_layout = result_layout or "rows" #"columns" # "rows" # "column-single" # "row-single"

    # In some sense, we can treat this as iteration start
    # TODO fix that we have two algorithms, add weights and fix algorithm_assignment (randomly assigning with each iteration)
    iteration_started(session["iteration"], [0.3,0.3,0.3], movies, {"0": "relevance", "1": "dummy"}, result_layout)

    return render_template("compare_algorithms.html", movies=movies, iteration=session["iteration"], result_layout=result_layout, MIN_ITERATION_TO_CANCEL=MIN_ITERATION_TO_CANCEL)

@bp.route("/refinement-feedback", methods=["GET"])
def refinement_feedback():
    version = request.args.get("version") or "1"
    return render_template("refinement_feedback.html", iteration=session["iteration"], version=version, metrics={"relevance": 70, "diversity": 20, "novelty": 10})


# We received feedback from compare_algorithms.html
@bp.route("/algorithm-feedback")
def algorithm_feedback():
    print("Inside algorithm feedback, forwarding to refinement_feedback")
    # TODO do whatever with the passed parameters and set session variable

    selected_movies = request.args.get("selected_movies")
    selected_movies = selected_movies.split(",") if selected_movies else []
    print(f"Selected movies={request.args.get('selected_movies')} after split={selected_movies}")
    selected_movies = [int(m) for m in selected_movies]
    x = session["selected_movie_indices"]
    x.append(selected_movies)
    session["selected_movie_indices"] = x
    print(f"Retrieved selected movies for iteration={session['iteration']} are: {session['selected_movie_indices']}")
    print(session["selected_movie_indices"])
    return redirect(url_for("plugin1.refinement_feedback"))

# We receive feedback from refinement_feedback.html
@bp.route("/refine-results")
def refine_results():
    # Go back to compare algorithms
    print("Inside refine results, increasing iteration and forwarding to compare_algorithms")
    session["iteration"] += 1
    # Generate new recommendations
    print(f"Selected movie indices = {session['selected_movie_indices']}")
    print(f"elicitation selected = {session['elicitation_selected_movies']}")
    print(f"movies={session['movies'][-1]}")
    mov = session["movies"]
    mov_indices = [[int(y["movie_idx"]) for y in x] for x in mov]
    filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], []) #sum(session["selected_movie_indices"], [])
    mov.append(recommend_2_3(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], filter_out_movies))
    session["movies"] = mov
    return redirect(url_for("plugin1.compare_algorithms"))

@bp.route("/final-questionare")
def final_questionare():
    if session["iteration"] < TOTAL_ITERATIONS:
        return render_template("final_questionare.html", iteration=session["iteration"], premature=True)
    return render_template("final_questionare.html", iteration=session["iteration"], premature=False)


def iteration_started(iteration, weights, movies, algorithm_assignment, result_layout):
    data = {
        "iteration": iteration,
        "weights": weights,
        "movies": movies,
        "algorithm_assignment": algorithm_assignment,
        "result_layout": result_layout
    }
    x = Interaction(
        participation = Participation.query.filter(Participation.id == session["participation_id"]).first(),
        interaction_type = "iteration-started",
        time = datetime.datetime.utcnow(),
        data = json.dumps(data)
    )
    db.session.add(x)


def iteration_ended(iteration, selected):
    data = {
        "iteration": iteration,
        "selected": selected
    }
    x = Interaction(
        participation = Participation.query.filter(Participation.id == session["participation_id"]).first(),
        interaction_type = "iteration-ended",
        time = datetime.datetime.utcnow(),
        data = json.dumps(data)
    )
    db.session.add(x)


@bp.route("/finish-user-study")
def finish_user_study():
    print("##########################@@@@@@@@@@@@@@@@@@@@")
    print(session["participation_id"])
    # print(Participation.query.filter(Participation.id == session["participation_id"]))
    Participation.query.filter(Participation.id == session["participation_id"]).update({"time_finished": datetime.datetime.utcnow()})
    db.session.commit()
    return render_template("finished_user_study.html")

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