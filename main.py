import streamlit as st
from fastai.vision.all import *
import plotly.express as xp
st.title("Medical Equiplment Classifier Model")

file = st.file_uploader("Upload image", type=['png', 'jpeg', 'gif'])

if file:

    img = PILImage.create(file)

    st.image(img)

    model = load_learner('medequipment_model2.pkl')

    prediction, pred_id, prob = model.predict(img)

    st.success(f"Prediction:{prediction}")
    st.info(f"Probability:{prob[pred_id]*100:.01f}%")

    figure = xp.bar(x = prob*100, y = model.dls.vocab)
    st.plotly_chart(figure)

