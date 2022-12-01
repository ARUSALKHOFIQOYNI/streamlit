from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np                       

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        with open('credit_risk_pickle_dt', 'rb') as r:
            model = pickle.load(r)

        pendapatan = float(request.form['pendapatan'])
        kpr_ya = float(request.form['kprya'])
        kpr_tidak = float(request.form['kprtidak'])
        pinjaman = float(request.form['pinjaman'])
        tanggungan = float(request.form['tanggungan'])
        overdue1 = float(request.form['overdue1'])
        overdue2 = float(request.form['overdue2'])
        overdue3 = float(request.form['overdue3'])
        overdue4 = float(request.form['overdue4'])
        overdue5 = float(request.form['overdue5'])

        datas = np.array((pendapatan,pinjaman,tanggungan,kpr_tidak,kpr_ya,overdue1,overdue2,overdue3,overdue4,overdue5))
        datas = np.reshape(datas, (1,10))
        credit_risk = model.predict(datas)

        return render_template('hasil.html', finalData=credit_risk)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)