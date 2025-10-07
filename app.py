from flask import Flask,redirect,render_template,send_file,session,request,url_for
import sys
import os
from ASH2.src.py.ash import ASH

app = Flask(__name__)
ash = ASH()

@app.route("/test")
def test():
    ash.run("how are you ash")


@app.route("/")
def index():
    return "A.S.H"  

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000 , debug=True)