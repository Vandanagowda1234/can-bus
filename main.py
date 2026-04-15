import pandas as pd

# Load dataset
data = pd.read_csv("data/dataset.csv", header=None)
data.columns = [
    "Timestamp", "ID", "DLC",
    "Data0", "Data1", "Data2", "Data3",
    "Data4", "Data5", "Data6", "Data7",
    "Label"
]
data = data.dropna()
# Convert ID
data['ID'] = data['ID'].apply(lambda x: int(str(x), 16))

# Convert Data columns
for col in ['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']:
    data[col] = data[col].apply(lambda x: int(str(x), 16))

# Show first 5 rows
print("First 5 rows:\n", data.head())

# Show column names
print("\nColumns:\n", data.columns)

# Show dataset shape
print("\nShape:", data.shape)

# Show data info
print("\nInfo:")
print(data.info())

# Check unique values in last column (usually label)
print("\nUnique values in last column:")
print(data.iloc[:, -1].unique())

# Separate input and output
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column

print("\nInput shape:", X.shape)
print("Output shape:", y.shape)

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

print("\nEncoded labels:", set(y))

# Keep only numeric data
X = X.select_dtypes(include=['number'])

print("\nAfter selecting numeric columns:", X.columns)

# Fill missing values
X = X.fillna(0)

print("\nFinal input shape:", X.shape)
print("Final output shape:", y.shape)

from sklearn.model_selection import train_test_split

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# create model
model = RandomForestClassifier()

# train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# predict on test data
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))