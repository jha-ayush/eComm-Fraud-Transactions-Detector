# eCommerce Fraud Transactions Detector

This codebase is a fraud detection model that identifies fraudulent transactions and compares different classification machine learning models for accuracy.

The csv data file contains the following columns with their respective datatypes:

- accountAgeDays -  `int64`
- numItems - `int64`
- localTime - `float64`
- paymentMethod - `object`
- paymentMethodAgeDays - `float64`
- label - `int64`


## Getting Started

To use this codebase, follow these steps:

- Create new conda environment `conda create -n [name] python=3.9`
- `conda activate [name]`
- `git clone` the repository to your local machine.
- Install the required packages using the following command: `pip install -r requirements.txt`
- Run the app.py file using the following command: `streamlit run home.py`
- Access the Streamlit web app at http://localhost:8501 in your web browser.


## Libraries Used
This codebase uses the following Python libraries:

- `streamlit` - for building the web app
- `pandas` - for data manipulation
- `scikit-learn` - for machine learning models
- `XGBoost` - for machine learning models
- `watermark` - for displaying the package versions
- `Warnings` - for suppressing warnings


## Usage

- Once you run the Streamlit app, you can use the following features:

- View the tail-end of the dataset and the data types.

- Encode the paymentMethod column using one-hot encoding and view the encoded data.

- Trigger the logistic regression classification and view the accuracy score and confusion matrix.

- Trigger the random forest classification and view the accuracy score and confusion matrix.


