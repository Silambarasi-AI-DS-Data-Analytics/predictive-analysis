import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_model(df):
    X = df[['Month']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    
    return model

def predict_future(model):
    future_months = pd.DataFrame({'Month': [49,50,51,52]})
    future_sales = model.predict(future_months)
    print("\nFuture Sales Prediction:")
    for i, sale in zip([49,50,51,52], future_sales):
        print(f"Month {i}: {sale:.0f} units")
    return future_sales