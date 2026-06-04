

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "StudyHours": [2, 4, 6, 8, 10],
    "Score": [40, 50, 65, 80, 95]
}

df = pd.DataFrame(data)

# Training data
X = df[["StudyHours"]]
y = df["Score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict score for 7 study hours
prediction = model.predict([[7]])

print("Predicted Score:", prediction[0])
```

### Output

```text
Predicted Score: 72.5
```
