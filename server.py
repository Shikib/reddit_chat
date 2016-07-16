from flask import Flask, render_template, redirect
from flask_socketio import SocketIO, join_room, leave_room, send, emit
import json
import bot
#Access a file /static/foo.html with the path /foo.html
app = Flask(__name__, static_url_path="")

cmb = None
def get_ai(subreddit):
    ai = bot.ContextAwareMarkovBot(ngram_len=10,
                                punctuation_dataset=subreddit,
                                style_dataset=subreddit,
                                subreddit=subreddit)
    ai.train_style()
    ai.train_punctuation()
    ai.reverse_train_style()
    ai.reverse_train_punctuation()
    return ai

 #Our routes
@app.route('/')
def root():
    return redirect("/r/AskReddit", code=302)

@app.route('/r/<subreddit>')
@app.route('/r/<subreddit>/<username>')
def get_subreddit(subreddit,username=None):
    return render_template("index.html", sub=subreddit, username=username)


app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)

@socketio.on('connect')
def handle_connection():
    print("user connected!")

@socketio.on('send_message')
def handle_myevent(data):
    emit("send_message",(data["message"], data["username"]), room=data["room"], include_self=False)
    emit("send_message",(subreddit_ai[data["room"]].generate_message(data["message"]), data["room"]+ " bot"), room=data["room"], broadcast=True)

@socketio.on('join_room')
def enter_user(data):
    join_room(data["room"])
    emit("send_message", (data["username"] + " has joined the room.", data["room"]+ " bot"), room=data["room"], broadcase=True);

if __name__ == '__main__':

    subreddit_ai =    {
            "AskReddit": get_ai("AskReddit"),
            "funny": get_ai("funny"),
            "WTF": get_ai("WTF"),
            "worldnews": get_ai("worldnews"),
            "videos": get_ai("videos"),
            "todayilearned": get_ai("todayilearned"),
            "science": get_ai("science"),
            "pics": get_ai("pics")
        }

    socketio.run(app, debug=True, port=8000)
