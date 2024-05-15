import numpy as np
import streamlit as st
import plotly.graph_objects as go

def updateLayout(fig,selected_model_type):

    fig.update_layout(
        title=f"Feature Importances for {selected_model_type}",
        xaxis_title="Importance",
        yaxis_title="Features",
        yaxis=dict(autorange="reversed")  # Invert the y-axis to show the most important feature at the top
    )

def graphLinearRegression(model, selected_model_type):
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

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_importances,
            y=sorted_feature_names,
            orientation='h',
            marker=dict(color=sorted_importances, colorscale='Greens'),
            textfont=dict(size=12, family='Arial Black'),
            text=[f"{imp:.2f}" for imp in sorted_importances],
            textposition='auto'
        )
    ])

    updateLayout(fig,selected_model_type)

    st.plotly_chart(fig)

def graphGradientBoostingRegressor(model, selected_model_type):
    # Get the feature importances from the model
    feature_importances = model.named_steps[selected_model_type.lower().replace("_", "")].feature_importances_

    # Sort the feature importances and feature names in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[sorted_indices]
    sorted_feature_names = [model.feature_names_in_[i] for i in sorted_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_importances,
            y=sorted_feature_names,
            orientation='h',
            marker=dict(color=sorted_importances, colorscale='Greens'),
            textfont=dict(size=12, family='Arial Black'),
            text=[f"{imp:.2f}" for imp in sorted_importances],
            textposition='auto'
        )
    ])
    
    updateLayout(fig, selected_model_type)
    st.plotly_chart(fig)

def graphSVR(model, selected_model_type, result):
    # Get the feature importances and their standard deviations
    importances = result.importances_mean
    std = result.importances_std

    # Sort the feature importances and feature names in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = [importances[i] for i in sorted_indices]
    sorted_feature_names = [model.feature_names_in_[i] for i in sorted_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_importances,
            y=sorted_feature_names,
            orientation='h',
            marker=dict(color=sorted_importances, colorscale='Greens'),
            textfont=dict(size=12, family='Arial Black'),
            text=[f"{imp:.2f} ± {std[i]:.2f}" for i, imp in zip(sorted_indices, sorted_importances)],
            textposition='auto'
        )
    ])
    
    updateLayout(fig, selected_model_type)
    st.plotly_chart(fig)

def graphLinearModels(model, selected_model_type):
    # Get the Random Forest model from the pipeline
    model_named_steps = model.named_steps[selected_model_type.lower().replace("_", "")]
    
    # Get the feature importances from the Random Forest model
    importances = model_named_steps.feature_importances_
    
    # Sort the feature importances and feature names in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = [importances[i] for i in sorted_indices]
    sorted_feature_names = [model.feature_names_in_[i] for i in sorted_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_importances,
            y=sorted_feature_names,
            orientation='h',
            marker=dict(color=sorted_importances, colorscale='Greens'),
            textfont=dict(size=12, family='Arial Black'),
            text=[f"{imp:.2f}" for imp in sorted_importances],
            textposition='auto'
        )
    ])
    
    updateLayout(fig, selected_model_type)
    st.plotly_chart(fig)