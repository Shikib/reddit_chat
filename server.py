from flask import Flask, render_template
from flask_socketio import SocketIO, join_room, leave_room, send, emit
import json
#Access a file /static/foo.html with the path /foo.html
app = Flask(__name__, static_url_path="")


 #Our routes
@app.route('/')
def root():
    return render_template("index.html")

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
    emit("send_message",data["message"], room=data["room"], include_self=False)

@socketio.on('join_room')
def enter_user(data):
    join_room(data["room"])
    emit("send_message", data["username"] + " has joined the room.", room=data["room"], broadcase=True);

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)
