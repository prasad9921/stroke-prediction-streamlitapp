import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

df=pd.read_csv("data/healthcare-dataset-stroke-data.csv")
# Define Features (Choosing 'avg_glucose_log')
# Target:'stroke'
X=df.drop(['stroke','avg_glucose_level'],axis=1) # Drop original glucose
Y=df['stroke']

# Identify Column Types
numeric_features=['age','hypertension','heart_disease','bmi','avg_glucose_log']
categorical_features=['gender','ever_married','work_type','Residence_type','smoking_status']

# Create Preprocessor
preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numeric_features),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ])

# Split Data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

# Fit & Transform
X_train_transformed=preprocessor.fit_transform(X_train)
X_test_transformed=preprocessor.transform(X_test)

# Save preprocessor for the app
joblib.dump(preprocessor,'preprocessor.joblib')
print("Preprocessing complete. Train/Test sets ready.")

import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,roc_auc_score,precision_score, 
                             recall_score,f1_score,matthews_corrcoef,confusion_matrix)

    

# Initialize the master metadata
model_metadata={"models":{}}

def update_metadata(name,params,metrics,history,path):
    model_metadata["models"][name]={
        "parameters":params,
        "final_metrics":metrics,
        "training_history":history,
        "model_path":path
    }
    with open('model_metadata.json','w') as f:
        json.dump(model_metadata,f,indent=4)

def plot_training_curve(history,title):
    plt.figure(figsize=(8,4))
    plt.plot(history['epoch'],history['train_acc'],label='Train Accuracy')
    if 'val_acc' in history:
        plt.plot(history['epoch'],history['val_acc'],label='Val Accuracy')
    plt.title(f'Training Curve: {title}')
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_metrics(y_true,y_pred,y_prob):
    return {
        "Accuracy":round(accuracy_score(y_true,y_pred),4),
        "AUC":round(roc_auc_score(y_true,y_prob),4),
        "Precision":round(precision_score(y_true,y_pred),4),
        "Recall":round(recall_score(y_true,y_pred),4),
        "F1":round(f1_score(y_true,y_pred),4),
        "MCC":round(matthews_corrcoef(y_true,y_pred),4)
    }

from sklearn.tree import DecisionTreeClassifier

model_name="Decision Tree"
history={'epoch':[],'train_acc':[]}

for i in range(1, 11):
    size=int((i/10)*len(X_train_transformed))
    clf=DecisionTreeClassifier(max_depth=7,random_state=42)
    clf.fit(X_train_transformed[:size],Y_train[:size])
    history['epoch'].append(i*10) # Represents % of data
    history['train_acc'].append(clf.score(X_train_transformed,Y_train))

y_pred=clf.predict(X_test_transformed)
y_prob=clf.predict_proba(X_test_transformed)[:,1]
metrics=get_metrics(Y_test,y_pred,y_prob)

plot_training_curve(history,f"{model_name}(Data Scaling)")
joblib.dump(clf,'decision_tree.joblib')
update_metadata(model_name,str(clf.get_params()),metrics,history,'decision_tree.joblib')