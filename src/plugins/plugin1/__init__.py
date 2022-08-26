# -*- coding: utf-8 -*-

from flask import Blueprint, jsonify, request, redirect, url_for, make_response

__plugin_name__ = "plugin1"
__version__ = "0.1.0"
__author__ = "staugur"

bp = Blueprint("plugin1", "plugin1")

@bp.route("/limit")
def limit():
    return "Hello from the plugin"
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