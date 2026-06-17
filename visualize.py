import matplotlib.pyplot as plt

def plot_predictions(df, future_sales):
    plt.figure(figsize=(10,6))
    plt.plot(df['Month'], df['Sales'], 
             label='Historical Sales', color='blue')
    plt.plot([49,50,51,52], future_sales, 
             label='Predicted Sales',
             color='red', linestyle='--', marker='o')
    plt.title('Sales Prediction Using Linear Regression')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()