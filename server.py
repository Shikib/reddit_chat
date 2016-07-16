from flask import Flask
import json
#Access a file /static/foo.html with the path /foo.html
app = Flask(__name__, static_url_path="")


 #Our routes
@app.route('/')
def root():
    return "hello world"



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
