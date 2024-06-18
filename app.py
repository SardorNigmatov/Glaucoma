import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px
import matplotlib.pyplot as plt
import platform
st.set_option('deprecation.showPyplotGlobalUse', False)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Glaucoma detection")


file = st.file_uploader("Rasmni yuklash", type=['png', 'jpeg', 'svg', 'jfif'])

if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)
    # model
    model = load_learner('./glaucoma_model.pkl')
    # Bashorat
    pred, pred_id, probs = model.predict(img)


    st.success(f"Bashorat qiymat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
 
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

    plt.figure()
    plt.bar(model.dls.vocab, probs)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(rotation=90)
    plt.title('Probability Distribution')

    st.pyplot()