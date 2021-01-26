import numpy as np
from sklearn.metrics import confusion_matrix

def cnn_model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)

    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:, 1]

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, \
        y_test_pred_proba, conf_matrix


def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D
    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

def evaluate_accuracies(predictions):
  actual=gsom_validation_labels
  # accuracy: (tp + tn) / (p + n)
  accuracy = accuracy_score(actual, predictions)
  print('Accuracy: %f' % accuracy)
  # precision tp / (tp + fp)
  precision = precision_score(actual, predictions)
  print('Precision: %f' % precision)
  # recall: tp / (tp + fn)
  recall = recall_score(actual, predictions)
  print('Recall: %f' % recall)
  # f1: 2 tp / (2 tp + fp + fn)
  f1 = f1_score(actual, predictions)
  print('F1 score: %f' % f1)

def overall_evaluation_test(p):
  real=[0,1,1,0,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0]  
  dep_correct=0
  dep_wrong=0
  norm_correct=0
  norm_wrong=0

  c=0

  for i in real:
    if i==0:
      if i==p[c]:
        norm_correct+=1
      else:
        norm_wrong+=1
    if i==1:
      if i==p[c]:
        dep_correct+=1
      else:
        dep_wrong+=1
    c+=1

  print("dep_correct: ",dep_correct)
  print("dep_wrong: ",dep_wrong)
  print("norm_correct: ",norm_correct)
  print("norm_wrong: ",norm_wrong)
  

def overall_evaluation_unbundle(p):
	test = [0,1,1,0,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0]
	unbundled_test_labels = []
	for i in test:
		if i==1:
			for j in range(46):
				unbundled_test_labels.append(1)
		else:
			for j in range(46):
				unbundled_test_labels.append(0)
				
	real=unbundled_test_labels  
	dep_correct=0
	dep_wrong=0
	norm_correct=0
	norm_wrong=0

	c=0

	for i in real:
		if i==0:
			if i==p[c]:
				norm_correct+=1
			else:
				norm_wrong+=1
		if i==1:
			if i==p[c]:
				dep_correct+=1
			else:
				dep_wrong+=1
		c+=1

	print("dep_correct: ",dep_correct)
	print("dep_wrong: ",dep_wrong)
	print("norm_correct: ",norm_correct)
	print("norm_wrong: ",norm_wrong)