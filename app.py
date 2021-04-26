from flask import Flask, render_template, redirect, request
from caption_it import caption_image

app=Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html")

@app.route('/', methods=["POST"])
def caption():
	if request.method=="POST":
		f=request.files['userfile']
		path="./static/{}".format(f.filename)
		f.save(path)
		generated_caption=caption_image(path)
		result_dict={
		'image': path,
		'caption': generated_caption
		}
		return render_template("index.html", result=result_dict)
	

@app.route('/home')
def redirect_to_home():
	return redirect('/')

if __name__ == '__main__':
	app.run(debug=True)