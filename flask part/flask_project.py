
from flask import Flask
app = Flask(__name__)
@app.route('/')
def home():
    return 'Hello, this is the home page!'
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask
app = Flask(__name__)
@app.route('/about')
def about():
    return 'this is about here, this is the about page!'
@app.route('/')
def home():
    return 'Hello, this is the home page!'
if __name__ == '__main__':
    app.run(debug=True)