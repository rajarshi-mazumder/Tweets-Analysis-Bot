from flask import Flask, render_template, request
import Twitter_Bot
from PIL import Image
from matplotlib import image
app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()
    name = output["name"]
    tweets_no= output["no_of_tweets"]
    img = image.imread("Messi.png")
    #print(type(img))
    Twitter_Bot.plot_graph(name, tweets_no)
    print(f'Username is {name} and No of tweets is {tweets_no}')
    return render_template("index.html", name=name, no_of_tweets= tweets_no, filename= img)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
