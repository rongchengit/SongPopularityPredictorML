import plotly.graph_objects as go
import streamlit as st

def graphModelFitting(model_names, x_value, y_value, metric):
    fig = go.Figure(data=[
        go.Bar(
            x=x_value,
            y=[model + '_' + metric for model in model_names],
            orientation='h',
               marker=dict(color='rgba(59, 245, 9, 0.8)'),
            text=[f"{val:.2f}" for val in x_value],
            textfont=dict(size=12, family='Arial Black'),
            textposition='auto',
            name=f'{metric} (Test)'
        ),
        go.Bar(
            x=y_value,
            y=[model + '_' + metric for model in model_names],
            orientation='h',
            marker=dict(color='rgba(0,68,27, 0.8)'),
            text=[f"{val:.2f}" for val in y_value],
            textfont=dict(size=12, family='Arial Black'),
            textposition='auto',
            name=f'{metric} (Training)'
        )
    ])

    fig.update_layout(
        title=f"Comparison of {metric} across Models",
        xaxis_title="Error",
        yaxis_title="Model",
        yaxis=dict(autorange="reversed"),  # Invert the y-axis to show the best model at the top
        barmode='group',
        legend=dict(x=1.02, y=1, orientation='v', bgcolor='rgba(255, 255, 255, 0.6)'),
        yaxis_tickmode='array',
        yaxis_tickvals=[model + '_' + metric for model in model_names],
        yaxis_ticktext=[model for model in model_names],
        height=500,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=100, r=200, t=100, b=50)
    )

    # Display the graph in Streamlit
    st.plotly_chart(fig)