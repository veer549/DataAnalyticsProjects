import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("traffic rules.csv")
print(df)
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns.to_list())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv("traffic_rules.csv", index=False)
df=pd.read_csv("traffic_rules.csv")

print("Missing values after cleaning:")
print(df.isnull().sum())

#EDA analysis
# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Average Speed'], bins=30, kde=True)
plt.title('Distribution of Average Speed')
plt.title('congestion level')
plt.xlabel('travel time index')
plt.ylabel('incident reoprts')
plt.legend()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Speed', y='Incident Reports', data=df)
plt.title('Incident Reports vs Average Speed')
plt.xlabel('Average Speed')
plt.ylabel('Incident Reports')
plt.legend()
plt.show()

# Multivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Speed', y='Incident Reports', hue='Congestion Level', data=df)
plt.title('Relationship between Average Speed and Incident Reports by Congestion Level')
plt.xlabel('Average Speed')
plt.ylabel('Incident Reports')
plt.legend()
plt.show()






# Linear Regression Model
X = df[['Average Speed', 'Congestion Level']]
y = df['Incident Reports']
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model training
model = LinearRegression()
model.fit(X_train, y_train)
# Prediction and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Predicted values: {y_pred}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Visualizing the regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Incident Reports')
plt.ylabel('Predicted Incident Reports')
plt.title('Actual vs Predicted Incident Reports')
plt.show()







