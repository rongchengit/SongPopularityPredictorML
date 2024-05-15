import streamlit as st
from streamlit_vertical_slider import vertical_slider

def graphFeatureImportance(model, sliders, audioFeatureRange):
    cols = st.columns(len(model.feature_names_in_) - 4)
    cols2 = st.columns(4)
    counter = 0
    for audioFeature in model.feature_names_in_:
        if audioFeature == 'explicit' or audioFeature == 'mode':
            # Boolean feature
            with cols2[2]:
                checkbox_value = st.checkbox(label=audioFeature)
                sliders[audioFeature] = 1 if checkbox_value else 0
        elif audioFeature == 'year':
            min_year = int(audioFeatureRange['year']['min'])
            max_year = int(audioFeatureRange['year']['max'])
            with cols2[1]:
                year_value = st.number_input(label='year', min_value=min_year, max_value=max_year, step=1, value=int((min_year + max_year) / 2))
            sliders[audioFeature] = year_value
        elif audioFeature == 'duration_ms':
            # Duration feature (converted to minutes and seconds)
            min_duration = int(audioFeatureRange['duration_ms']['min'] / 1000)  # Convert milliseconds to seconds
            max_duration = int(audioFeatureRange['duration_ms']['max'] / 1000)  # Convert milliseconds to seconds
            min_minutes, min_seconds = divmod(min_duration, 60)
            max_minutes, max_seconds = divmod(max_duration, 60)
            with cols2[0]:
                duration_str = st.text_input(
                    label="Song Duration",
                    value= '3:00'
                )
            if duration_str:
                try:
                    duration_parts = duration_str.split(':')
                    duration_minutes = int(duration_parts[0])
                    duration_seconds = int(duration_parts[1])
                    if (duration_minutes < min_minutes or
                        (duration_minutes == min_minutes and duration_seconds < min_seconds) or
                        duration_minutes > max_minutes or
                        (duration_minutes == max_minutes and duration_seconds > max_seconds)):
                        st.warning('Please enter a duration within the valid range.')
                    else:
                        duration_ms = (duration_minutes * 60 + duration_seconds) * 1000  # Convert back to milliseconds
                        sliders[audioFeature] = duration_ms
                except (ValueError, IndexError):
                    st.warning('Please enter the duration in the format MM:SS.')
            else:
                st.warning('Please enter a duration.')
        else:
            # Numeric feature
            if audioFeature == 'key' or audioFeature == 'time_signature' or audioFeature == 'genre' or audioFeature == 'artist':
                min_value = int(audioFeatureRange[audioFeature]['min'])
                max_value = int(audioFeatureRange[audioFeature]['max'])
                median_value = round((min_value + max_value) / 2)
                with cols[counter]:
                    counter = counter + 1
                    ing_slider = vertical_slider(
                        label=audioFeature,
                        min_value=min_value,
                        max_value=max_value,
                        default_value=median_value,
                        step=1,
                        height=300,
                        thumb_shape="circle",
                        track_color="green",
                        slider_color='rgba(228, 224, 216)',
                        thumb_color="rgba(59, 245, 9)",
                        value_always_visible=True
                    )
            else:
                min_value = float(audioFeatureRange[audioFeature]['min'])
                max_value = float(audioFeatureRange[audioFeature]['max'])
                median_value = (min_value + max_value) / 2
                with cols[counter]:
                    counter = counter + 1
                    ing_slider = vertical_slider(
                        label=audioFeature,
                        min_value=min_value,
                        max_value=max_value,
                        step = (max_value-min_value) / 50,
                        default_value=median_value,
                        height=300,
                        thumb_shape="circle",
                        track_color="green",
                        slider_color='rgba(228, 224, 216)',
                        thumb_color="rgba(59, 245, 9)",
                        value_always_visible=True
                    )
            sliders[audioFeature] = ing_slider

def graphRandomForestClassifierConfidence(model, sliders):
    probs = model.predict_proba([sliders])
    probability = probs[0][1]  # Assuming the positive class is at index 1
    st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))