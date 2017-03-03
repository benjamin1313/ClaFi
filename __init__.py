# -*- coding: UTF-8 -*-
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug import secure_filename
import tensorflow as tf, sys 
import os
app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg','jpeg'])
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg']) 
app.config['UPLOAD_FOLDER'] = "/path/to/Static"
#print("stage1")
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS 
@app.route('/',methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
            # check if the post request has the file part
            out = "post "
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also submit a empty 
            # part without filename
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
                #print("stage2")
                    
                return redirect(url_for('result', filename=filename))
                
                                        
    body=Markup('''<center><div style="width:50%;">
		<h1>Upload et billede af en fisk</h1>
		<form action="" method=post enctype=multipart/form-data>
			<table>
			<tr>
				<td align="left"><input type=file name=file></td>
				<td align="right"><input type=submit value=Upload></td>
			<tr>
			</table>
		</form>
		<br><br>
		<div class="alert alert-info"><strong>Info!!!</strong> Dit billede skal vaere i jpg format ellers kan jeg ikke laese det</div>
		<div class="alert alert-info"><strong>Info!!!</strong> Naar du uploader et billede giver du mig tilladelse til at gemme og bruge det til at laere</div>
	</div></center>''')
    return render_template("menu.html",body=body)
    
    
def predict(image_path):
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.gfile.GFile("/home/benjamin1313/fisk/FlaskApp/tf_files/retrained_labels.txt")]
    
    #print("stage4")
    # Unpersists graph from file
    with tf.gfile.FastGFile("/home/benjamin1313/fisk/FlaskApp/tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    #print("stage5")
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first 
        # prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    return top_k, label_lines, predictions 


@app.route('/about/')
def about():
    body = Markup('''<center><div style="width:50%;"><h1>Hej jeg er ClaFi</h1>
    <p>Jeg er en kunsting intelegens, lavet til at genkende fisk via billeder, Jeg er baseret paa <a href="http://www.tensorflow.org/">TensorFlow</a>
    Og jeg er traenet paa 3961 billeder af 11 forskelige fisk.</p>
    <br>
    <br>
    <p>Jeg bliver vedligeholdt af benjamin1313 og er hosted paa en maskine hos <a href="http://nerdhosting.dk">Nerdhosting</a></p>
    </center>''')
    return render_template("menu.html",body=body)

    
@app.route('/result/',methods=['GET', 'POST'])
def result():
    filename = request.args.get('filename')
    image_path = "/home/benjamin1313/fisk/FlaskApp/static/"+str(filename)
    out = ''' <center><img src="'''+url_for('static',filename=filename)+'''"width="400"><br/>resultaterne for dette billede: <br/>'''
    #print("stage3")
    try:
        top_k, label_lines, predictions = predict(image_path)
        #print("stage6")
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            out = out + str(human_string) + " " + str(score*100) + "%<br/>"
            #print('%s (score = %.5f) ' % (human_string, score))
            body=out + "</center>"
            body=Markup(body)
        return render_template('menu.html', body=body)
    except:
        return "<center>Beklager men der opstod et problem med behandlingen af dit billede, proev igen</center>"

if __name__ == '__main__':
    app.run()
