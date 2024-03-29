import numpy as np
import streamlit as st
import scikitplot as skplt

def graphLinearRegression(model, selected_model_type, ax1, feat_imp_fig):
    # Get the linear classifier model from the pipeline
    model_named_steps = model.named_steps[selected_model_type.lower().replace("_", "")]
    # Get the coefficients of the linear classifier
    coefficients = model_named_steps.coef_

    # Check if coefficients is a single value or an array
    if isinstance(coefficients, np.float64):
        # If it's a single value, convert it to a list
        coefficients = [coefficients]
    else:
        # If it's an array, flatten it to a 1D list
        coefficients = coefficients.flatten().tolist()

    # Calculate the absolute values of the coefficients
    abs_coefficients = [abs(coef) for coef in coefficients]

    # Normalize the absolute coefficients to get the feature importances
    feature_importances = [coef / sum(abs_coefficients) for coef in abs_coefficients]

    # Sort the feature importances and feature names in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_importances = [feature_importances[i] for i in sorted_indices]
    sorted_feature_names = [model.feature_names_in_[i] for i in sorted_indices]

    # Create a bar plot of the feature importances
    ax1.bar(range(len(sorted_importances)), sorted_importances)

    # Set the x-tick labels to the feature names
    ax1.set_xticks(range(len(sorted_importances)))
    ax1.set_xticklabels(sorted_feature_names, rotation=70)

    # Set the plot title and labels
    ax1.set_title("Feature Importances")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Importance")

    st.pyplot(feat_imp_fig, use_container_width=True)

def graphGradientBoostingRegressor(model, ax1, feat_imp_fig):
    # Get the feature importances from the model
    feature_importances = model.feature_importances_

    # Sort the feature importances and feature names in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[sorted_indices]
    sorted_feature_names = [model.feature_names_in_[i] for i in sorted_indices]

    # Create a bar plot of the sorted feature importances
    ax1.bar(range(len(sorted_importances)), sorted_importances)

    # Set the x-tick labels to the sorted feature names
    ax1.set_xticks(range(len(sorted_importances)))
    ax1.set_xticklabels(sorted_feature_names, rotation=70)

    # Set the plot title and labels
    ax1.set_title("Feature Importances")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Importance")

    st.pyplot(feat_imp_fig, use_container_width=True)

def graphLinearModels(model, selected_model_type, ax1, feat_imp_fig):
    model_named_steps = model.named_steps[selected_model_type.lower().replace("_", "")]
    skplt.estimators.plot_feature_importances(model_named_steps, 
                                            feature_names=model.feature_names_in_, 
                                            ax=ax1, x_tick_rotation=70)
    st.pyplot(feat_imp_fig, use_container_width=True)