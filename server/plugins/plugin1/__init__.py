# -*- coding: utf-8 -*-

import datetime
import json
import os
import time
from flask import Blueprint, jsonify, request, redirect, url_for, make_response, render_template, session

from plugins.utils.preference_elicitation import recommend_2_3, rlprop, weighted_average, result_layout_variants

from models import Interaction, Participation
from app import db, pm
from common import get_tr, load_languages, multi_lang
import glob

import numpy as np
import random

__plugin_name__ = "plugin1"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

NUM_TO_SELECT = 5

MIN_ITERATION_TO_CANCEL = 5
TOTAL_ITERATIONS = 8

HIDE_LAST_K = 1000000 # Effectively hides everything

languages = load_languages(os.path.dirname(__file__))
print(f"Languages={languages}")

# Map internal algorithm names to those displayed to user
algorithm_name_mapping = {
    #"rlprop": "beta",
    "relevance_based": "gamma",
    "weighted_average": "delta"
}

# Implementation of this function can differ among plugins
def get_lang():
    default_lang = "en"
    if "lang" in session and session["lang"] and session["lang"] in languages:
        return session["lang"]
    return default_lang


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

# Public facing endpoint
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
    print("After preference elicitation")
    return render_template("step.html", step_number=1, movies=session["movies"][-1])

@bp.route("/compare-algorithms", methods=["GET"])
def compare_algorithms():

    k_param = request.args.get("k")
    if k_param:
        x = session["selected_movie_indices"]
        if not x:
            x.append([])
            session["selected_movie_indices"] = x
            prepare_recommendations(k=int(k_param))
            session["selected_movie_indices"] = x[:-1]
        else:
            prepare_recommendations(k=int(k_param))
    # movies = [session["movies"][-1]]
    # movies.append([movies[0][0]] * len(movies[0]))
    algorithm_assignment = {}
    algorithms = list(algorithm_name_mapping.keys())
    # Make them in random order so that they are displayed differently
    random.shuffle(algorithms)
    movies = {}
    for i, algorithm in enumerate(algorithms):
        if session["movies"][algorithm][-1]:
            # Only non-empty makes it to the results
            movies[algorithm_name_mapping[algorithm]] = session["movies"][algorithm][-1]
            algorithm_assignment[str(i)] = {
                "algorithm": algorithm,
                "name": algorithm_name_mapping[algorithm]
            }



    result_layout = request.args.get("result_layout") or "rows"
    #result_layout = result_layout or "rows" #"columns" # "rows" # "column-single" # "row-single"
    #result_layout = result_layout_variants[session["permutation"][0]]
    
    # Decide on next refinement layout
    refinement_layout = "3" # Use version 3
    session["refinement_layout"] = refinement_layout

    # In some sense, we can treat this as iteration start
    # TODO fix that we have two algorithms, add weights and fix algorithm_assignment (randomly assigning with each iteration)
    iteration_started(session["iteration"], session["weights"], movies, algorithm_assignment, result_layout, refinement_layout)

    params = {
        "movies": movies,
        "iteration": session["iteration"],
        "result_layout": result_layout,
        "MIN_ITERATION_TO_CANCEL": len(result_layout_variants)
    }

    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliary_university"] = tr("footer_cagliary_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("compare_title")
    params["header"] = tr("compare_header")
    params["note"] = tr("note")
    params["algorithm"] = tr("algorithm")
    params["note_text"] = tr("compare_note_text")
    params["hint"] = tr("compare_hint")
    params["hint_lead"] = tr("compare_hint_lead")
    params["algorithm_satisfaction"] = tr("compare_algorithm_satisfaction")
    params["like_nothing"] = tr("compare_like_nothing")
    params["significantly"] = tr("compare_significantly")
    params["slightly"] = tr("compare_slightly")
    params["same"] = tr("compare_same")
    params["next"] = tr("next")
    params["finish"] = tr("compare_finish")

    return render_template("compare_algorithms.html", **params)

@bp.route("/refinement-feedback", methods=["GET"])
def refinement_feedback():
    version = request.args.get("version") or session["refinement_layout"] #"1"
    return render_template("refinement_feedback.html", iteration=session["iteration"], version=version,
        metrics={
            "relevance": session["weights"][0],
            "diversity": session["weights"][1],
            "novelty": session["weights"][2]
        }
    )


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
    
    # return redirect(url_for("plugin1.refinement_feedback")) # TODO uncomment for main user study
    ##### START OF NEW, SHORTER VERSION
    # Since we never get to refine-results, we have to move some of the stuff here
    # E.g. we should call iteration ended here, weights are kept the same
    iteration_ended(session["iteration"] - 1, session["selected_movie_indices"], session["weights"])    
    # Increase iteration
    session["iteration"] += 1
    # And generate new recommendations
    prepare_recommendations(k=session["rec_k"])
    # And shift the permutation
    permutation = session["permutation"]
    permutation = permutation[1:] + permutation[:1] # Move first item to the end
    session["permutation"] = permutation
    return redirect(url_for("plugin1.compare_algorithms"))
    ##### END OF NEW, SHORTER VERSION

def prepare_recommendations(k=10):
    mov = session["movies"]

    # Randomly chose two algorithms
    algorithms = ["relevance_based", "weighted_average"]
    assert len(mov[algorithms[0]]) == len(mov[algorithms[1]]), "All algorithms should share the number of iterations"
    
    for algorithm in algorithms:
        mov[algorithm].append([])

    mov_indices = []
    for i in range(len(mov[algorithms[0]])):
        indices = set()
        for algo in algorithms:
            indices.update([int(y["movie_idx"]) for y in mov[algo][i]])
        mov_indices.append(list(indices))

    
    filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], [])

    # Always generate recommendation via relevance based algorithm because we need to get the model (we use it as a baseline)
    recommended_items, model = recommend_2_3(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], filter_out_movies, return_model=True, k=k)    

    # Order of insertion should be preserved
    for algorithm in algorithms:
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], model, np.array(session['weights']), filter_out_movies, k=k)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], model, np.array(session['weights']), filter_out_movies, k=k)
        else:
            assert False
        mov[algorithm][-1] = recommended_items

    session["movies"] = mov

# We receive feedback from refinement_feedback.html
@bp.route("/refine-results")
def refine_results():
    # Get new weights
    new_weights = request.args.get("new_weights")
    new_weights = [float(x) for x in new_weights.split(",")]
    session["weights"] = new_weights

    # Go back to compare algorithms
    print("Inside refine results, increasing iteration and forwarding to compare_algorithms")
    session["iteration"] += 1
    # Generate new recommendations
    print(f"Selected movie indices = {session['selected_movie_indices']}")
    print(f"elicitation selected = {session['elicitation_selected_movies']}")
    #print(f"movies={session['movies'][-1]}")
    mov = session["movies"]
    
    # Randomly chose two algorithms
    algorithms = ["relevance_based", "weighted_average"]
    assert len(mov[algorithms[0]]) == len(mov[algorithms[1]]), "All algorithms should share the number of iterations"
    
    for algorithm in algorithms:
        mov[algorithm].append([])

    mov_indices = []
    for i in range(len(mov[algorithms[0]])):
        indices = set()
        for algo in algorithms:
            indices.update([int(y["movie_idx"]) for y in mov[algo][i]])
        mov_indices.append(list(indices))

    
    filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], [])

    # Always generate recommendation via relevance based algorithm because we need to get the model (we use it as a baseline)
    recommended_items, model = recommend_2_3(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], filter_out_movies, return_model=True)    

    # Order of insertion should be preserved
    for algorithm in algorithms:
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], model, np.array(new_weights), filter_out_movies)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], model, np.array(new_weights), filter_out_movies)
        else:
            assert False
        mov[algorithm][-1] = recommended_items
    # END NEW

    # This worked
    #mov_indices = [[int(y["movie_idx"]) for y in x] for x in mov]
    #filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], []) #sum(session["selected_movie_indices"], [])
    #mov.append(recommend_2_3(session["elicitation_selected_movies"] + session["selected_movie_indices"][-1], filter_out_movies))
    
    session["movies"] = mov




    # In some sense, session ended here
    iteration_ended(session["iteration"] - 1, session["selected_movie_indices"], new_weights)    

    return redirect(url_for("plugin1.compare_algorithms"))

@bp.route("/final-questionare")
@multi_lang
def final_questionare():
    params = {
        "iteration": session["iteration"]
    }

    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliary_university"] = tr("footer_cagliary_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("final_title")
    params["header"] = tr("final_header")
    params["finish"] = tr("final_finish")
    params["hint"] = tr("final_hint")

    params["premature"] = False
    if session["iteration"] < TOTAL_ITERATIONS:
        params["premature"] = True

    return render_template("final_questionare.html", **params)


def iteration_started(iteration, weights, movies, algorithm_assignment, result_layout, refinement_layout):
    data = {
        "iteration": iteration,
        "weights": weights,
        "movies": movies,
        "algorithm_assignment": algorithm_assignment,
        "result_layout": result_layout,
        "refinement_layout": refinement_layout
    }
    x = Interaction(
        participation = Participation.query.filter(Participation.id == session["participation_id"]).first().id,
        interaction_type = "iteration-started",
        time = datetime.datetime.utcnow(),
        data = json.dumps(data)
    )
    db.session.add(x)
    db.session.commit()


def iteration_ended(iteration, selected, new_weights):
    data = {
        "iteration": iteration,
        "selected": selected,
        "new_weights": new_weights
    }
    x = Interaction(
        participation = Participation.query.filter(Participation.id == session["participation_id"]).first().id,
        interaction_type = "iteration-ended",
        time = datetime.datetime.utcnow(),
        data = json.dumps(data)
    )
    db.session.add(x)
    db.session.commit()


@bp.route("/finish-user-study")
@multi_lang
def finish_user_study():
    print("##########################@@@@@@@@@@@@@@@@@@@@")
    print(session["participation_id"])
    # print(Participation.query.filter(Participation.id == session["participation_id"]))
    Participation.query.filter(Participation.id == session["participation_id"]).update({"time_finished": datetime.datetime.utcnow()})
    db.session.commit()

    params = {}
    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliary_university"] = tr("footer_cagliary_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("finish_title")
    params["header"] = tr("finish_header")
    params["hint"] = tr("finish_hint")

    return render_template("finished_user_study.html", **params)

@bp.route("/step2", methods=["GET"])
def step2():
    return render_template("step.html", step_number=2)

@bp.route("/preference-elicitation", methods=["GET", "POST"])
@multi_lang
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