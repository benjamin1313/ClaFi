from flask import Flask, request, redirect, url_for, render_template
from werkzeug import secure_filename
import tensorflow as tf, sys
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = "static"
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
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #print("stage2")
                    
                return redirect(url_for('result', filename=filename))
                                        
    return '''<!doctype html>
    <title>Fisk classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <center>
    <h1>Upload et billede af en fisk</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    <p>tilladte filer: png og jpg</p>
    <p>ved at uploade et billede giver du mig tilladelse til at gemme det til saa jeg kan blive bedre.</p>
    </center>'''
    
    
def predict(image_path):
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]
    
    #print("stage4")
    # Unpersists graph from file
    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    #print("stage5")
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    return top_k, label_lines, predictions
    
@app.route('/result/',methods=['GET', 'POST'])
def result():
    filename = request.args.get('filename')
    image_path = "static/"+str(filename)
    out = '''
    <center>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <img src="'''+url_for('static',filename=filename)+'''" width="400"><br/>
    resultaterne for dette billede: <br/>
    '''
    #print("stage3")
    #try:
    top_k, label_lines, predictions = predict(image_path)
    #print("stage6")
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        out = out + str(human_string) + " " + str(score*100) + "%<br/>"
        #print('%s (score = %.5f) ' % (human_string, score))
            
    return out + "</center>"
    #except:
     #   return "<center>Beklager men der opstod et problem med behandlingen af dit billede, proev igen</center>"
if __name__ == '__main__':
    app.run()

