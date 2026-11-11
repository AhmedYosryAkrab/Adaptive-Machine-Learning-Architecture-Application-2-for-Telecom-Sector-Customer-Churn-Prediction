# =============================================
# Step 1: Imports and Initial Setup
# =============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc)
from imblearn.over_sampling import BorderlineSMOTE

# =============================================
# Step 1.1: Load and Preprocess Data(Data preprocessing phase)
# =============================================
df = pd.read_csv('churn_prediction_clean.csv')

nominal_columns = df.select_dtypes(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder()
df[nominal_columns] = ordinal_encoder.fit_transform(df[nominal_columns])

imputer = SimpleImputer(strategy='median')
df[df.select_dtypes(include=['float64', 'int64']).columns] = \
    imputer.fit_transform(df[df.select_dtypes(include=['float64', 'int64']).columns])

df[nominal_columns] = df[nominal_columns].fillna(-1)

X_data = df.drop(['Churn'], axis=1)
y = df['Churn']

# Apply BorderlineSMOTE
smote = BorderlineSMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_data, y)

features = pd.DataFrame(X_resampled, columns=X_data.columns)
churn = y_resampled

X_train, X_test, y_train, y_test = train_test_split(features, churn, test_size=0.3, random_state=0)

# =============================================
# Step 2: Feature Selection Using L-BSSA(Feature selection phase)
# =============================================
class Spider:
    def __init__(self, position, fitness):
        self.position = position
        self.fitness = fitness

def initialize_population(search_space, features, churn):
    population = []
    for _ in range(num_spiders):
        position = np.random.randint(2, size=search_space)
        fitness = evaluate_fitness(position, features, churn)
        population.append(Spider(position, fitness))
    return population

def update_position(spider, best_neighbor, transfer_func):
    difference = spider.position - best_neighbor.position
    update_value = transfer_func(difference)
    if np.random.rand() < 0.5:
        spider.position = np.where(spider.position != best_neighbor.position, update_value, spider.position)

def transfer_function(difference):
    return np.where(difference != 0, np.random.randint(2, size=difference.shape), difference)

def evaluate_fitness(position, features, churn):
    selected_indices = [i for i in range(len(position)) if position[i] == 1]
    if not selected_indices:
        return 0

    selected_features_data = features.iloc[:, selected_indices]
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(selected_features_data, churn, test_size=0.3, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=5, random_state=42)
    model.fit(X_train_split, y_train_split)
    y_pred = model.predict(X_val_split)
    fitness = accuracy_score(y_val_split, y_pred)
    return fitness

def L_BSSA(search_space, num_spiders, max_iterations, features, churn):
    population = initialize_population(search_space, features, churn)
    for _ in range(max_iterations):
        best_spider = max(population, key=lambda spider: spider.fitness)
        for spider in population:
            neighbors = np.random.choice(population, size=2, replace=False)
            best_neighbor = max(neighbors, key=lambda neighbor: neighbor.fitness)
            update_position(spider, best_neighbor, transfer_function)
        for spider in population:
            spider.fitness = evaluate_fitness(spider.position, features, churn)
    return max(population, key=lambda spider: spider.fitness)

search_space = features.shape[1]
num_spiders = 8
max_iterations = 15

best_spider = L_BSSA(search_space, num_spiders, max_iterations, features, churn)

selected_features_indices = np.where(best_spider.position == 1)[0]
top_features = [features.columns[i] for i in selected_features_indices]

print("Top selected features:", top_features)

# =============================================
# Step 3: Model Training and Evaluation(Learning phase)
# =============================================
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Train Random Forest with depth and regularization
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
rf_classifier.fit(X_train_selected, y_train)
# =============================================
# Step 4: Predict and Evaluate(Performance evaluation phase)
# =============================================
y_pred = rf_classifier.predict(X_test_selected)
y_proba = rf_classifier.predict_proba(X_test_selected)[:, 1]

print("\n=== Model Performance Using LBSA-selected Features ===")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Learning Curve #Cross validation applied
train_sizes, train_scores, test_scores = learning_curve(
    rf_classifier, X_train_selected, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, label='Validation Score', color='green', marker='s')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.title('Learning Curve (Random Forest with LBSA Features)')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Feature Importance
importance_df = pd.DataFrame({
    'Feature': X_train_selected.columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()