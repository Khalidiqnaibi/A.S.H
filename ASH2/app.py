from flask import Flask,redirect,render_template,send_file,session,request,url_for,jsonify
from src.py.ash import ASH ,USER
from .utils.kio import *

app = Flask(__name__)
ash = ASH()

@app.route("/")
def index():
    return "A.S.H is running"

@app.route("/q/<q>")
def q(q):
    query = input(f"{USER}: ")
    kinput(query,by=USER)
    response = ash.run(query)
    return(jsonify(response))