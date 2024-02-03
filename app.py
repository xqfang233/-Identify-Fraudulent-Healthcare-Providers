from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('xgboost_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':

        In_ClaimID = float(request.form['In_ClaimID'])
        In_BeneID=float(request.form['In_BeneID'])
        In_claim_duration=float(request.form['In_claim_duration'])
        In_DeductibleAmtPaid=float(request.form['In_DeductibleAmtPaid'])
        Ot_12Months_PartBCov=float(request.form['Ot_12Months_PartBCov'])
        In_InscClaimAmtReimbursed=float(request.form['In_InscClaimAmtReimbursed'])
        Ot_ClmDiagnosisNum = float(request.form['Ot_ClmDiagnosisNum'])
        Ot_12Months_PartACov = float(request.form['Ot_12Months_PartACov'])

        dp = [In_ClaimID, In_BeneID, In_claim_duration, In_DeductibleAmtPaid,
                                  Ot_12Months_PartBCov, In_InscClaimAmtReimbursed, Ot_ClmDiagnosisNum,
                                  Ot_12Months_PartACov]

        prediction=model.predict([dp])
        predict_prob = model.predict_proba([dp])
        res=round(prediction[0],2)

        if res=="0":
            return render_template('index.html',prediction_texts="This is not a fraudulent provider. The probability of being non-fraudulent is {}, the probaility of being fraudulent is {}.".format(predict_prob[:,0],predict_prob[:,1]))
        elif res=="1":
            return render_template('index.html',prediction_text="You caught a fraudulent provider!!! The probability of being non-fraudulent is {}, the probaility of being fraudulent is {}.".format(predict_prob[:,0],predict_prob[:,1]))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

