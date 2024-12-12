# Streamlit App Code with Explanation
# Streamlit is an open-source app framework designed to help developers create and share
# beautiful, interactive web applications for machine learning and data science projects quickly.
# It's easy to use and requires minimal coding effort to transform a Python script into a functional web app.

# Import the required libraries
import streamlit as st

# Set the title of the app
st.title("Simple Streamlit App")

# Explanation about Streamlit
st.write("""
### What is Streamlit?
Streamlit is an open-source Python library that makes it easy to build custom web apps for data visualization,
data analysis, and machine learning. It allows you to create interactive user interfaces with just a few lines of code.
Streamlit apps run in your browser and are ideal for showcasing projects, prototypes, or dashboards.
""")

# Input from the user
name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=0, step=1)

# Display the input back to the user
if st.button("Submit"):
    st.write(f"Hello, {name}! You are {age} years old.")

# Simple visualization example
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Sine Wave")
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")

# Display the plot in the app
st.pyplot(fig)
