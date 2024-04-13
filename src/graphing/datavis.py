import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def generate_plot(df, feature):
    # Calculate number of songs and outliers
    num_songs = len(df[feature])
    if not feature == 'explicit':
        q1, q3 = np.percentile(df[feature], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        num_outliers = len(df[feature][(df[feature] < lower_bound) | (df[feature] > upper_bound)])
        outlier_percentage = (num_outliers / num_songs) * 100
    else:
        outlier_percentage = 0
    # Create subplots
    fig = make_subplots(rows=1, cols=3, column_widths=[0.6, 0.2, 0.2], subplot_titles=("Boxplot", "Scatterplot", "Histogram"))

    # Boxplot
    fig.add_trace(go.Box(y=df[feature], name=feature, showlegend=False, boxpoints=False), row=1, col=1)

    # Scatterplot
    fig.add_trace(go.Scatter(x=np.random.normal(0, 0.05, size=len(df[feature])), y=df[feature], mode='markers', marker=dict(size=5, opacity=0.5), showlegend=False), row=1, col=2)

    # Histogram
    hist, bins = np.histogram(df[feature], bins=20)
    colors = ['rgba(255, 0, 0, {})'.format(opacity) for opacity in np.linspace(0.2, 1, len(hist))]
    fig.add_trace(go.Bar(x=[0.5] * len(hist), y=hist, width=0.8, marker=dict(color=colors), orientation='v', showlegend=False), row=1, col=3)

    # Text
    fig.update_layout(
        title=f"Analysis of {feature}",
        annotations=[
            dict(
                x=0.5,
                y=1.15,
                xref="paper",
                yref="paper",
                text=f"Number of Songs: {num_songs}",
                showarrow=False,
                font=dict(size=12),
                align="center"
            ),
            dict(
                x=0.5,
                y=1.10,
                xref="paper",
                yref="paper",
                text=f"Outlier Percentage: {outlier_percentage:.2f}%",
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
        ]
    )

    # Update axis properties
    fig.update_yaxes(title_text=feature, row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=3)

    # Display the figure
    st.plotly_chart(fig)

def is_outlier(data, feature):
  Q1 = data[feature].quantile(0.25)
  Q3 = data[feature].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - (1.5 * IQR)
  upper_bound = Q3 + (1.5 * IQR)
  return (data[feature] < lower_bound) | (data[feature] > upper_bound)

def generateBarChart(valueScore, modelName, label):
    fig = go.Figure(data=[
            go.Bar(
                x=valueScore,
                y=modelName,
                orientation='h',
                marker=dict(color=valueScore, colorscale='Viridis'),
                text=[f"{val:.2f}" for val in valueScore],
                textposition='auto',
                name='MSE'
            )
        ])

    fig.update_layout(
            title=f"Comparison of {label} across Models",
            xaxis_title=f"{label}",
            yaxis_title="Model",
            yaxis=dict(autorange="reversed")  # Invert the y-axis to show the best model at the top
        )

    # Display the graph in Streamlit
    st.plotly_chart(fig)

def graphCorrelations(correlations):
    fig = go.Figure(data=go.Heatmap(
        z=correlations.values,
        x=correlations.columns,
        y=correlations.index,
        colorscale='Viridis',
        zmin=-1,  # Set the minimum value of the color scale to -1
        zmax=1    # Set the maximum value of the color scale to 1
    ))

    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        xaxis_tickangle=-45  # Rotate the x-axis labels by -45 degrees for better readability
    )

    st.plotly_chart(fig)

def graphPopularityCorrelations(correlations):
    # Get the correlations with the 'popularity' column
    popularity_correlations = correlations['popularity'].drop('popularity')
    
    # Sort the correlations in descending order
    sorted_correlations = popularity_correlations.sort_values(ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        x=sorted_correlations.values,
        y=sorted_correlations.index,
        orientation='h',
        marker=dict(
            color=sorted_correlations.values,
            colorscale='Viridis',
            colorbar=dict(title='Correlation')
        )
    )])
    
    fig.update_layout(
        title="Correlations with Popularity",
        xaxis_title="Correlation",
        yaxis_title="Features"
    )
    
    st.plotly_chart(fig)

