# Import Libraries
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score # Accuracy score

from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


#------------------------------------------------------------------#

# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="Crypto Predictor",page_icon="💳",layout="centered",initial_sidebar_state="auto")

@st.cache_data # Add cache data decorator

# Load and Use local style.css file
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")


#------------------------------------------------------------------#

# Read ticker symbols from a CSV file
try:
    df = pd.read_csv("./Resources/data.csv")
except:
    logging.error('Cannot find the CSV file')
    
    

st.header("eCommerce Fraud Detector") # Title
st.write("Fraudulent transactions identification and comparison against different Classification Machine Learning models")
st.write("---")
st.write(f"<b>Displaying tail-end of the data</b>",unsafe_allow_html=True)
st.write(df.tail(100)) # Display data tail
st.write(f"<b>Displaying data types</b>",unsafe_allow_html=True)
st.write(df.dtypes)

# Encode Payment methods using one-hot encoding
df_encoded = pd.get_dummies(df, columns=["paymentMethod"])


# Apply one-hot encoding to paymentMethod column
if st.button("View Payment method encoded data"):
    st.write(f"<b>Displaying tail-end of the encoded data</b>",unsafe_allow_html=True)
    st.write(df_encoded.tail(100)) # Display encoded data tail




# Split dataset up into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded.drop('label', axis=1), df_encoded['label'],
    test_size=0.30, random_state=1)

## Binary Classification - Logistic Regression

# Create a button to trigger the prediction
if st.button("Logistic Regression ML classification model"):

    # Create hyperparameter grid for Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 1000, 2500, 5000],
        'class_weight': ['balanced', None]
    }

    # Create RandomizedSearchCV object
    clf = LogisticRegression(random_state=1)
    random_clf = RandomizedSearchCV(clf, param_grid, n_iter=50, cv=5, random_state=1, n_jobs=-1)

    # Fit the RandomizedSearchCV object to the training data
    random_clf.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = random_clf.predict(X_test)

    st.write(f"<b>Accuracy score:</b> ",accuracy_score(y_pred, y_test),unsafe_allow_html=True)


    ## Evaluating the Fraud Detection Model ##

    # Compare test set predictions with ground truth labels - Confusion Matrix
    st.write(f"<b>Logistic Regression - Confusion Matrix Algorithm</b>",unsafe_allow_html=True)
    st.write(confusion_matrix(y_test, y_pred))
    st.info(f"we can observe in the Confusion Matrix, using the Logistic Regression Machine Learning model, that 165 transactions are recognized as 'Fraudulent' transactions. 11,602 are recognized as 'Not Fraudulent' transactions.")



## Random Forest Classification

# Create a button to trigger the prediction
if st.button("Random Forest ML classification model"):

    # Create hyperparameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 300, 500, 800, 1200],
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15, 100],
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False]
    }


    # Create RandomizedSearchCV object
    rf_clf = RandomForestClassifier(random_state=1)
    random_rf_clf = RandomizedSearchCV(rf_clf, rf_param_grid, n_iter=50, cv=5, random_state=1, n_jobs=-1)

    # Fit the RandomizedSearchCV object to the training data
    random_rf_clf.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = random_rf_clf.predict(X_test)

    st.write(f"<b>Accuracy score:</b> ",accuracy_score(y_pred, y_test),unsafe_allow_html=True)

    ## Evaluating the Fraud Detection Model ##

    # Compare test set predictions with ground truth labels - Confusion Matrix
    st.write(f"<b>Random Forest - Confusion Matrix Algorithm</b>",unsafe_allow_html=True)
    st.write(confusion_matrix(y_test, y_pred))
    st.info(f"From the Confusion Matrix, using the Random Forest Machine Learning model, we can observe that 165 transactions are recognized as 'Fraudulent' transactions. 11,602 are recognized as 'Not Fraudulent' transactions.")



## XGBoost Classification

# Create a button to trigger the prediction
if st.button("XGBoost ML classification model"):
    
    # Create hyperparameter grid for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 300, 500, 800, 1200],
        'max_depth': [5, 10, 15, 20, 30, None],
        'learning_rate': [0.01, 0.1, 0.5, 1],
        'subsample': [0.5, 0.7, 1],
        'colsample_bytree': [0.5, 0.7, 1],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.1, 0.5, 1],
        'scale_pos_weight': [1, 2, 5, 10]
    }


    # Create an instance of the XGBoost classifier
    xgb_clf = XGBClassifier(random_state=1)


    # Create a RandomizedSearchCV object for XGBoost
    random_xgb_clf = RandomizedSearchCV(xgb_clf, xgb_param_grid, n_iter=50, cv=5, random_state=1, n_jobs=-1)

    # Fit the RandomizedSearchCV object to the training data
    random_xgb_clf.fit(X_train, y_train)

    # Make Predictions on the test set
    y_pred = random_xgb_clf.predict(X_test)


    # Evaluate the accuracy of the XGBoost model
    st.write(f"<b>Accuracy score:</b> ",accuracy_score(y_pred, y_test),unsafe_allow_html=True)

    ## Evaluate the XGBoost model using the confusion matrix ##

    # Compare test set predictions with ground truth labels - Confusion Matrix
    st.write(f"<b>XGBoost - Confusion Matrix Algorithm</b>",unsafe_allow_html=True)
    st.write(confusion_matrix(y_test, y_pred))
    st.info(f"From the Confusion Matrix, using the XGBoost Machine Learning model, we can observe that 165 transactions are recognized as 'Fraudulent' transactions. 11,602 are recognized as 'Not Fraudulent' transactions.")

