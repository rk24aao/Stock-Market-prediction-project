# Stock Market Prediction Using Historical Apple Stock Data

## Project Overview

This project was completed as part of my MSc Data Science course at the University of Hertfordshire.

The main purpose of this project is to investigate whether historical stock market data can be used to predict the next-day closing price of Apple (AAPL) stock. Three different forecasting models were developed and compared:

- ARIMA
- XGBoost
- LSTM

The performance of the models was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

## Research Question

How effectively can ARIMA, XGBoost and LSTM predict the next-day closing price of Apple stock using historical market data, and which approach performs best on unseen time-ordered test data?

## Aim

The aim of this project is to develop and compare ARIMA, XGBoost and LSTM models for predicting the next-day closing price of Apple stock using historical market data.

## Dataset

Historical Apple (AAPL) stock data was collected using the `yfinance` Python library.

The original dataset contains the following variables:

- Date
- Open
- High
- Low
- Close
- Volume

The data was cleaned and prepared before performing exploratory data analysis and model development.

The target variable was created by shifting the closing price by one trading day:

```python
stock_data['Target'] = stock_data['Close'].shift(-1)
```

Therefore, the models use historical information to predict the closing price of the next trading day.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the characteristics and patterns in the Apple stock dataset.

The analysis included:

- Checking the structure of the dataset
- Checking missing values
- Descriptive statistics
- Apple closing-price trend
- Open and Close price comparison
- High and Low price comparison
- Trading volume analysis
- Daily return analysis
- 20-day moving average (MA20)
- 50-day moving average (MA50)
- Distribution of closing prices
- Correlation heatmap

The correlation heatmap showed strong relationships between the main stock price variables such as Open, High, Low and Close. Moving averages were also strongly related to the closing price, while variables such as Daily Return and Volume showed weaker relationships.

## Feature Engineering

Additional features were created from the historical stock data to support the machine-learning models.

These included:

- Daily Return
- MA20
- MA50
- Lag1
- Lag2
- Lag3
- Lag5
- Lag10
- EMA10
- EMA20
- RSI
- MACD
- Signal Line

The exact features used depended on the requirements of each model.

## Models

### 1. ARIMA

ARIMA was used as the traditional statistical time-series forecasting model.

The ARIMA model mainly used historical closing-price information to predict the next closing price. A chronological train-test split was used and walk-forward forecasting was performed on the test data.

The baseline model used an ARIMA (3,1,0) configuration.

Hyperparameter selection was also performed to identify another suitable ARIMA order. The tuned model selected an ARIMA (5,2,0) configuration.

### 2. XGBoost

XGBoost was used as the machine-learning model.

Unlike ARIMA, XGBoost used several historical price variables, lag features and technical indicators as input features.

The data was split chronologically rather than randomly because this is a time-series prediction problem.

The baseline XGBoost model was trained first. Hyperparameter tuning was then performed using `GridSearchCV` with `TimeSeriesSplit`.

The parameters considered during tuning included:

- Number of estimators
- Learning rate
- Maximum depth
- Subsample
- Column sampling

The best tuned configuration included 400 estimators, a learning rate of 0.1, a maximum depth of 5, and subsample and column sampling values of 0.8.

### 3. LSTM

LSTM was used as the deep-learning model because it is designed to learn sequential patterns in time-series data.

The LSTM model used the previous 60 trading days as an input sequence to predict the next-day closing price.

The data was scaled using `MinMaxScaler`, with the scaler fitted using the training data. The baseline LSTM contained 64 LSTM units followed by a dropout layer and an output layer.

Hyperparameter tuning was performed using Keras Tuner RandomSearch to test different combinations of:

- LSTM units
- Dropout rate
- Learning rate

The tuned model was then evaluated on the unseen test data.

## Model Evaluation

The models were evaluated using three regression metrics:

**Mean Absolute Error (MAE)**  
Measures the average absolute difference between the actual and predicted prices. A lower MAE indicates better performance.

**Root Mean Squared Error (RMSE)**  
Measures prediction error while giving greater importance to larger errors. A lower RMSE indicates better performance.

**R-squared (R²)**  
Measures how well the model explains variation in the target variable. A value closer to 1 generally indicates better predictive performance.

## Results

The final results of the baseline and tuned models were:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| ARIMA Baseline | 2.13 | 2.84 | 0.9907 |
| ARIMA Tuned | 2.33 | 3.07 | 0.9891 |
| XGBoost Baseline | 22.09 | 31.79 | -0.4120 |
| XGBoost Tuned | 23.07 | 32.63 | -0.4873 |
| LSTM Baseline | 5.06 | 6.03 | 0.958 |
| LSTM Tuned | 6.72 | 7.44 | 0.936 |

## Main Findings

The baseline ARIMA model achieved the best overall performance, with the lowest MAE and RMSE and the highest R².

LSTM also performed well and was able to learn useful sequential patterns from the historical stock data. However, it did not outperform ARIMA.

XGBoost produced the weakest performance. One possible reason was that the later test period contained higher Apple stock prices than much of the training period. This made it difficult for the tree-based model to generalise to the later price range.

Another important finding was that hyperparameter tuning did not improve the final test performance of any of the three models. The baseline configurations performed better than their corresponding tuned models on the unseen test data.

This shows that a more complex model or additional hyperparameter tuning does not always result in better stock-price predictions.

## Project Files

The main files included in this repository are:

```text
Stock-Market-prediction-project/
│
├── Finance data set download and EDA.ipynb
├── ARIMA.ipynb
├── xg-boost.ipynb
├── LSTM.ipynb
├── aapl_cleaned.csv
├── README.md
└── .gitignore
```

### Finance data set download and EDA.ipynb

Contains:

- Apple stock data collection
- Data cleaning
- Feature engineering
- Exploratory Data Analysis
- Visualisations
- Correlation analysis

### ARIMA.ipynb

Contains:

- ARIMA baseline model
- Walk-forward forecasting
- Hyperparameter selection
- Tuned ARIMA model
- Model evaluation
- Actual vs predicted price visualisation

### xg-boost.ipynb

Contains:

- XGBoost feature preparation
- Chronological train-test split
- Baseline XGBoost model
- GridSearchCV
- TimeSeriesSplit
- Hyperparameter tuning
- Feature importance
- Model evaluation

### LSTM.ipynb

Contains:

- Data scaling
- 60-day sequence creation
- Baseline LSTM model
- Keras Tuner RandomSearch
- Hyperparameter tuning
- Model evaluation
- Actual vs predicted price visualisation

### aapl_cleaned.csv

Contains the cleaned Apple stock dataset used for model development.

## Technologies Used

The project was developed using Python and Jupyter Notebook.

Main Python libraries used include:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- pmdarima
- XGBoost
- TensorFlow
- Keras
- Keras Tuner
- yfinance

## How to Run the Project

1. Clone or download this repository.

2. Install the required Python libraries.

3. Open the notebooks using Jupyter Notebook, JupyterLab, VS Code, or another compatible environment.

4. Run the EDA notebook first:

```text
Finance data set download and EDA.ipynb
```

5. The modelling notebooks can then be run separately:

```text
ARIMA.ipynb
xg-boost.ipynb
LSTM.ipynb
```

The exact results may vary slightly depending on library versions and the training process of the LSTM model.

## Limitations

This project has several limitations.

The study focuses only on Apple stock, so the findings cannot automatically be generalised to other companies or financial markets.

The models mainly use historical market information. External factors such as financial news, company announcements, economic conditions and investor sentiment were not included.

The project also predicts the next-day closing-price level. Future studies could investigate more challenging targets such as next-day returns or whether the stock price will increase or decrease.

## Future Work

The project could be extended by:

- Testing the models on other companies
- Using a longer or different historical period
- Predicting stock returns
- Predicting price direction
- Including financial news sentiment
- Including macroeconomic indicators
- Testing additional machine-learning and deep-learning models
- Applying additional walk-forward validation approaches

## Ethical Considerations

The project uses historical financial market data and does not contain personal or sensitive information. No human participants, surveys, interviews or social-media data were used.

The data was used for academic and educational purposes. Model results, including unsuccessful hyperparameter-tuning results, were reported to provide a transparent comparison of the forecasting approaches.

## Disclaimer

This project was completed for academic and educational purposes only.

The predictions produced by the models should not be considered financial or investment advice. Stock prices are affected by many factors that are not included in this project, and past market behaviour does not guarantee future performance.

## Author

**Ramya Sree Koka**  
MSc Data Science  
University of Hertfordshire