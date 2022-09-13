# -*- coding: utf-8 -*-

import os
from flask import Blueprint, jsonify, request, redirect, url_for, make_response, render_template

__plugin_name__ = "plugin1"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint("plugin1", "plugin1")

NUM_TO_SELECT = 5


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