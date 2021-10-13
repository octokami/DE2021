# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import logging
import os

import pandas as pd
import numpy as np

from flask import jsonify
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score as r2

def train(dataset):
    X = dataset[['Acres', 'Deck', 'GaragCap', 'Patio', 'PkgSpacs', 'Taxes', 'TotBed', 'TotBth', 'TotSqf']]
    y = dataset['SoldPrice']
    X = sm.add_constant(X)

    r2_vals = []

    for i in range(10) :

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        model = sm.OLS(y_train, X_train).fit()
        model_pred = model.predict(X_test)
        r2_vals.append(r2(y_test, model_pred))

    print('Mean R-Squared over 10 runs: ' + str(np.mean(r2_vals))) 
    print("")
    text_out = {'Mean R-Squared over 10 runs:' :str(np.mean(r2_vals))}
    #text_out = {model.summary()}

    ridge_reg = Ridge(alpha=10, fit_intercept=True)
    ridge_reg.fit(X_train, y_train)

    cols = ['const','Acres', 'Deck', 'GaragCap', 'Patio', 'PkgSpacs', 'Taxes', 'TotBed', 'TotBth', 'TotSqf']
    print("Ridge regression model:\n {}+ {}^T . X".format(ridge_reg.intercept_, ridge_reg.coef_))
    pd.Series(ridge_reg.coef_.flatten(), index=cols)

    
    # Saving model in a given location provided as an env. variable
    model_repo = os.environ['MODEL_REPO']
    model = ridge_reg
    
    if model_repo:
        file_path = os.path.join(model_repo, "model.h5")
        #model.save(file_path)
        model = pickle.dumps(model, file_path)
        logging.info("Saved the model to the location : " + model_repo)
        return jsonify(text_out), 200
    else:
        #model.save("model.h5")
        model = pickle.dumps(model, "model.pkl")
        return jsonify({'message': 'The model was saved locally.'}), 200
