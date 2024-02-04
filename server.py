import pickle


# Function that takes loads in our pickled word processor
# and defines a function for using it. This makes it easy
# to do these steps together when serving our model.
def predict_fraud(dp):
    model = pickle.load(open('xgboost_model.pkl', 'rb'))

    # Function to apply our model & extract keywords from a
    # provided bit of text
    # define predictions
    prediction = model.predict([dp])
    pred_prob = model.predict_proba([dp])
    res = prediction[0]
    if res == 0:
        out_text = ("This is not a fraudulent provider. The probability "
                    "of being non-fraudulent is {}, the probaility of being "
                    "fraudulent is {}.").format(pred_prob[:,0],pred_prob[:,1])
    elif res==1:
        out_text = ("You caught a fraudulent provider!!! The probability "
                    "of being non-fraudulent is {}, the probaility of being "
                    "fraudulent is {}.").format(pred_prob[:,0],pred_prob[:,1])


    # return the function we just defined
    return out_text