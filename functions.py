import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def display_model_metrics(y_test, y_pred, y_pred_proba, model_name='Model'):
    """
    Displays the ROC curve and confusion matrix heatmap for a given model.
    
    Parameters:
        y_test (array-like): True labels of the test set.
        y_pred (array-like): Predicted labels of the test set.
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model (default is 'Model').
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Plot the ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} ROC Curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='No Skill', alpha=0.7)  # Diagonal line
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

    # Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'{model_name} Confusion Matrix Heatmap', fontsize=16)
    plt.show()

def display_instance_explanation(model, X_test, y_test, index_of_interest, X_test_scaled=None, feature_names=None):
    """
    Display the actual label, predicted label, feature values, and SHAP values for a specific instance.

    Parameters:
        model: Trained model to explain.
        X_test (DataFrame or ndarray): Test features.
        y_test (Series or ndarray): Test labels.
        index_of_interest (int): Index of the instance to explain.
        X_test_scaled (ndarray, optional): Scaled test features (if applicable).
        feature_names (list, optional): List of feature names for readability.
    """
    # Check if the index is valid
    if index_of_interest in y_test.index:
        # Extract the scaled feature values if provided, otherwise use X_test directly
        X_instance = (X_test_scaled[index_of_interest].reshape(1, -1)
                      if X_test_scaled is not None else X_test.iloc[index_of_interest].values.reshape(1, -1))
        y_instance = y_test.iloc[index_of_interest]  # Actual label for the instance

        # Get predicted label
        y_pred_instance = model.predict(X_instance)[0]

        # Print the actual and predicted labels
        print(f"Instance Index: {index_of_interest}")
        print(f"Actual Label: {y_instance}")
        print(f"Predicted Label: {y_pred_instance}")

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for the specific instance
        shap_values_instance = explainer.shap_values(X_instance)

        # Prepare feature names if not provided
        if feature_names is None:
            feature_names = X_test.columns if isinstance(X_test, pd.DataFrame) else [f"Feature {i}" for i in range(X_test.shape[1])]

        # Convert SHAP values into a DataFrame for better readability
        shap_values_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values_instance[0],
            'Feature Value': X_test.iloc[index_of_interest].values
        })

        # Print the SHAP values DataFrame sorted by absolute SHAP value (importance)
        print("\nSHAP Values (sorted by importance):")
        print(shap_values_df.sort_values(by='SHAP Value', key=abs, ascending=False))

        # Visualize the SHAP values using a force plot
        print("\nSHAP Force Plot:")
        shap.force_plot(
            explainer.expected_value,
            shap_values_instance[0],
            X_instance,
            feature_names=feature_names,
            matplotlib=True
        )
    else:
        print(f"Index {index_of_interest} is not valid in the test dataset.")