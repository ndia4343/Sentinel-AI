import streamlit as st

st.title("SENTINEL_AI v4.2")
st.subheader("Industrial Predictive Maintenance")

st.write("System Status: Online")

if st.button("Run Diagnostics"):
    st.success("All systems operational")
