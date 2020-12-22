from flask import Flask,request,jsonify
from app.torch_util_funcs import transform_image,prediction
app=Flask(__name__)

def allowed_file(filename):
    ALLOWED_EXT={'png','jpg','jpeg'}
    return '.' in filename and filename.split('.')[1].lower() in ALLOWED_EXT

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=="POST":
        file=request.files.get('file')
        # print(file.filename,allowed_file(file.filename))
        if file is None or file.filename=="":
            return jsonify({"error":"No File Found"})
        if not allowed_file(file.filename):
            return jsonify({"error":"File Format Not Supported"})
        try:
            img_data=file.read()
            img_tensor=transform_image(img_data)
            pred=prediction(img_tensor)
            data={'prediction':pred[0].item(),'class_name':str(pred[0].item())}
            return jsonify(data)
        except:
            return jsonify({"error":"Unexpected Error During Prediction"})
    return jsonify({'result':1})
