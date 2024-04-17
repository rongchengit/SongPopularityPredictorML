import math
import pandas as pd
import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def generate_scatterplot(df, feature):
    # Assuming you have a DataFrame named 'df' with columns 'loudness' and 'popularity'
    fig = px.scatter(df, x=feature, y='popularity', color='popularity', color_continuous_scale='Greens', title=f'Correlation between {feature} and Popularity')

    # Update the axis labels
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title='Popularity')

    # Display the plot
    st.plotly_chart(fig)

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
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.2], subplot_titles=("Boxplot", "Heatmap"))

    # Boxplot
    fig.add_trace(go.Box(y=df[feature], name=feature, showlegend=False, boxpoints=False, marker=dict(color='green')), row=1, col=1)

    # Heatmap
    min_val = df[feature].min()
    max_val = df[feature].max()
    bins = np.linspace(min_val, max_val, 10)
    hist, _ = np.histogram(df[feature], bins=bins)
    hist_normalized = hist / hist.max()

    #fig.add_trace(go.Heatmap(x=np.ones(len(bins)-1),
    #                        y=bins[:-1],
    #                        z=hist.reshape(-1, 1),
    #                        colorscale='Greens',
    #                        showscale=True,
    #                        colorbar=dict(title='Count')),
    #            row=1, col=2)

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
                marker=dict(color=valueScore, colorscale='Greens'),
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
        colorscale='Green',
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
            colorscale='Greens',
            colorbar=dict(title='Correlation')
        )
    )])
    
    fig.update_layout(
        title="Correlations with Popularity",
        xaxis_title="Correlation",
        yaxis_title="Features"
    )
    
    st.plotly_chart(fig)

def generate_report(df, feature):   
    # Step 1: Calculate min and max values for each feature
    feature_min = math.floor(df[feature].min())
    feature_max = math.ceil(df[feature].max())

    # Step 2: Calculate bin size for each feature
    bin_size = (feature_max - feature_min) / 10

    if bin_size == 0:
        bin_size = 1

    # Step 3: Create bins for each feature
    bins = np.arange(feature_min, feature_max + bin_size, bin_size)

    # Step 4: Count the number of songs in each bin for each feature
    bin_counts = pd.cut(df[feature], bins, include_lowest=True).value_counts().sort_index()

    # Step 5: Create a report DataFrame
    report_data = []
    for i, count in enumerate(bin_counts):
        bin_start = bins[i]
        bin_end = bins[i+1]
        startmin, startsec = divmod(int(bin_start) / 1000, 60)
        endmin, endsec = divmod(int(bin_end) / 1000, 60)
        report_data.append([f"{round(startmin)}:{round(startsec)} - {round(endmin)}:{format(round(endsec), '02d')}", count])

    report_df = pd.DataFrame(report_data, columns=["Bin Range", "Song Count"])

    # Step 6: Display the report in Streamlit
    st.markdown("**Song Count per Bin**")
    st.write(report_df)