from data import get_data
from model import train_model, predict_future
from visualize import plot_predictions

# Step 1: Load Data
print("Loading dataset...")
df = get_data()
print(f"Dataset loaded! Total records: {len(df)}")

# Step 2: Train Model
print("\nTraining model...")
model = train_model(df)

# Step 3: Predict Future
print("\nPredicting future sales...")
future_sales = predict_future(model)

# Step 4: Visualize
print("\nGenerating chart...")
plot_predictions(df, future_sales)

print("\nProject Complete! ")