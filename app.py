from flask import Flask,redirect,render_template,url_for,request,flash
import pickle
import pandas as pd

app = Flask(__name__)

# ============================================== Loading Model =============================================

model = pickle.load(open('models/model.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

# ================================================= Routes =================================================

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/talent_acquisition_system')
def talent_acquisition_system():
    return render_template('job.html')

# =========================================== Prediction Function ==========================================

def prediction(gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    data = {
    'gender': [gender],
    'ssc_p': [ssc_p],
    'hsc_p': [hsc_p],
    'degree_p': [degree_p],
    'workex': [workex],
    'etest_p': [etest_p],
    'specialisation': [specialisation],
    'mba_p': [mba_p]
    }
    data = pd.DataFrame(data)
    data['gender'] = data['gender'].map({'Male':1,"Female":0})
    data['workex'] = data['workex'].map({"Yes":1,"No":0})
    data['specialisation'] = data['specialisation'].map({"Mkt&HR":1,"Mkt&Fin":0})
    scaled_df = scaler.transform(data)
    result = model.predict(scaled_df).reshape(1, -1)
    return result[0]

# =============================================== Prediction ===============================================

@app.route('/placement',methods=['POST','GET'])
def pred():
    if request.method == 'POST':
        gender = request.form['gender']
        ssc_p = request.form['ssc_p']
        hsc_p = request.form['hsc_p']
        degree_p = request.form['degree_p']
        workex = request.form['workex']
        etest_p = request.form['etest_p']
        specialisation = request.form['specialisation']
        mba_p = request.form['mba_p']

        result = prediction(gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p)

        if result == 1:
            pred = "Placed"
            rec = "This candidate fits well with your business goals."
            return render_template('job.html', result=pred, rec=rec)

        else:
            pred = "Not Placed"
            rec = "This candidate may not align with the requirements of your business."
            return render_template('job.html', result=pred,rec=rec)

    return redirect(url_for('index'))

# ============================================== Python Main ==============================================

if __name__ == "__main__":
    app.run(debug=True)