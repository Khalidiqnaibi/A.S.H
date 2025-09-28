from flask import Flask,redirect,render_template,send_file,session,request,url_for


app = Flask(__name__)


@app.route("/")
def index():
    return "A.S.H"