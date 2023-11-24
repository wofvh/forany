#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[3]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score


# In[ ]:





# In[5]:





# In[6]:


# Load the actual and predicted labels from CSV files
# actual_df = pd.read_csv("H:/Contil/Research/MobileScaffloding/classify/classify/groundtruth/GT.csv", header=None)
# predicted_df = pd.read_csv("H:/Contil/Research/MobileScaffloding/classify/classify/prediction/output.csv", header=None)

actual_df = pd.read_csv("C:\\f1score\\test\\GroundTruth\\GT.csv", header=None)
predicted_df = pd.read_csv("C:\\f1score\\test\\Prediction\\PT.csv", header=None)

# Extract the labels as arrays
y_true = actual_df.iloc[:, 1].values
y_pred = predicted_df.iloc[:, 1].values

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Compute the classification report
cr = classification_report(y_true, y_pred)
print("Classification Report:\n", cr)

# Compute the accuracy
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)

# Compute the precision
prec = precision_score(y_true, y_pred)
print("Precision:", prec)

# Compute the recall
rec = recall_score(y_true, y_pred)
print("Recall:", rec)

# Compute the F1-score
f1 = f1_score(y_true, y_pred)


# In[7]:


from sklearn.metrics import roc_auc_score, roc_curve

auc_score = micro_roc_auc_ovr = roc_auc_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)


# In[5]:


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[34]:


auc_score


# In[19]:


from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    y_true[:,],
    y_pred)


# In[6]:


# Compute the ROC
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()


# In[ ]:





# In[ ]:




