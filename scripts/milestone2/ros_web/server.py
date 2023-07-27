import os
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/my-link/')
def my_link():
  print ('Backend: click received!')

  return 'Click.'

if __name__ == '__main__':
  app.run(debug=True)

# Use this to make it accessible from the internet(aka your phone)
# http://localhost.run/docs/