from flask import Flask,redirect,render_template,send_file,session,request,url_for,jsonify
import sys
import os
from ASH2.src.py.ash import ASH , USER
# from ASH2.utils.kio import *

app = Flask(__name__)
ash = ASH()

@app.route("/test")
def test():
    response = ash.run("how are you ash")
    return(jsonify(response))

@app.route("/q/<q>")
def q(q):
    query = f"{USER}: {q}"
    # kinput(query,by=USER)
    response = ash.run(query)
    return(jsonify(response))

@app.route("/")
def index():
    return "A.S.H"  

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000 , debug=True)