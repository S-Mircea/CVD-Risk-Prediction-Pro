from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <html>
    <head><title>Test Flask App</title></head>
    <body>
        <h1>🎉 Flask is Working!</h1>
        <p>If you can see this, Flask is running properly.</p>
        <p>Your CVD app should work too!</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting test Flask app...")
    print("Open: http://localhost:9000")
    app.run(host='localhost', port=9000, debug=False)