import os
from dotenv import load_dotenv
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load .env variables
load_dotenv()

# Paths
DATA_PATH = './irisdataset.csv'  # For local testing, use relative path
MODEL_PATH = './model.pkl'

# Read environment variables safely
class_label_env = os.getenv('CLASS_LABEL')
kernel_env = os.getenv('KERNEL')

if class_label_env is None:
    raise ValueError("Environment variable 'CLASS_LABEL' is not set.")
if kernel_env is None:
    raise ValueError("Environment variable 'KERNEL' is not set.")

try:
    CLASS_LABEL = int(class_label_env)
except ValueError:
    raise ValueError("Environment variable 'CLASS_LABEL' must be an integer.")

KERNEL = kernel_env

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

# Split into train/test sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# Train model
model = svm.SVC(C=CLASS_LABEL, kernel=KERNEL)
model.fit(train_x, train_y)

# Evaluate model
score = model.score(test_x, test_y)
print(f"Test Data Score: {score}")

# Save trained model
pickle.dump(model, open(MODEL_PATH, 'wb'))
