import streamlit as st
import pandas as pd
from models import KNN, DecisionTree, RandomForest
from models.preprocessing import treat_input_data
import plotly.express as px
from sklearn.decomposition import PCA
from models.utils import Strategy
from typing import get_args

st.title("Predicting Soil Sample Classification 😎")

# Load your preprocessed data
data = pd.read_csv('data/Dataset1.csv')
X, y, scalar = treat_input_data(data, normalization='minmax')

if "fitted" not in st.session_state:
    st.session_state.fitted = False


@st.cache_data
def reduce_dimensions(x):
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)
    return x_reduced


# Initialize models
models = {
    'KNN': {
        'class': KNN,
        'type': 'supervised',
        'params': {
            'k': {
                'min': 1,
                'max': 50,
                'default': 5,
                'key': 'k',
                'label': 'n_neighbors',
                'type': 'slider'
            },
            'strategy': {
                'default': get_args(Strategy)[0],
                'options': get_args(Strategy),
                'key': 'strategy',
                'label': 'strategy',
                'type': 'select'
            }
        }
    },
    'Decision Tree': {
        'class': DecisionTree,
        'type': 'supervised',
        'params': {
            'max_depth': {
                'min': 1,
                'max': 100,
                'default': 50,
                'key': 'max_depth',
                'label': 'max_depth',
                'type': 'slider'
            }
        }
    },
    'Random Forest': {
        'class': RandomForest,
        'type': 'supervised',
        'params': {
            'n_trees': {
                'min': 1,
                'max': 10,
                'default': 5,
                'key': 'n_trees',
                'label': 'n_trees',
                'type': 'slider'
            },
            'max_depth': {
                'min': 1,
                'max': 10,
                'default': 50,
                'key': 'max_depth',
                'label': 'max_depth',
                'type': 'slider'
            }
        }
    }
}

# Streamlit App
st.title("Soil Sample Classification App")

# Model selection
selected_model = st.selectbox("Select Model", list(models.keys()), on_change=st.session_state.clear)

# Params for each model

st.write(f"Selected Model: {selected_model}")

# Display model parameters
st.subheader("Model Parameters:")
for param in models[selected_model]['params'].values():
    match param['type']:
        case 'slider':
            st.slider(label=param.get('label'), min_value=param.get('min'), max_value=param.get('max'),
                      value=param.get('default'), key=param.get('key'))
        case 'select':
            st.selectbox(label=param.get('label'), options=param.get('options'), key=param.get('key'))

# Model instantiation
model = models[selected_model]
model_instance = model['class'](**{param: st.session_state[param] for param in model['params']})

# Input parameters
st.header("Input Parameters")

user_data = {
    'N': None,
    'P': None,
    'K': None,
    'pH': None,
    'EC': None,
    'OC': None,
    'S': None,
    'Zn': None,
    'Fe': None,
    'Cu': None,
    'Mn': None,
    'B': None,
    'OM': None
}

st.subheader("User Input:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    user_data['N'] = st.number_input("Enter a value for N")
    user_data['P'] = st.number_input("Enter a value for P")
    user_data['K'] = st.number_input("Enter a value for K")

with col2:
    user_data['pH'] = st.number_input("Enter a value for pH")
    user_data['EC'] = st.number_input("Enter a value for EC")
    user_data['OC'] = st.number_input("Enter a value for OC")

with col3:
    user_data['S'] = st.number_input("Enter a value for S")
    user_data['Zn'] = st.number_input("Enter a value for Zn")
    user_data['Fe'] = st.number_input("Enter a value for Fe")
with col4:
    user_data['Cu'] = st.number_input("Enter a value for Cu")
    user_data['Mn'] = st.number_input("Enter a value for Mn")
    user_data['B'] = st.number_input("Enter a value for B")

user_data['OM'] = st.number_input("Enter a value for OM")

# Buttons for fitting and prediction
fit_button = st.button("Fit Model")
predict_button = st.button("Predict")

if fit_button:
    model_instance.fit(X, y)
    st.session_state['model'] = model_instance
    st.success(f"{selected_model} has been fitted with the data.")
    if 'fig' in st.session_state:
        del st.session_state['fig']
        del st.session_state['fig2']
    st.session_state.fitted = True

if predict_button:
    # Handle missing values in input_data if needed
    if None in user_data.values():
        st.warning("Please provide values for all input parameters.")

    if not st.session_state.fitted:
        st.error("Please fit the model first.")
    else:
        input_data = [user_data[param] for param in user_data]
        input_data = scalar.transform([input_data])
        predictions = st.session_state['model'].predict_single(input_data)
        st.write(f"Prediction is: {predictions}")

if st.session_state.fitted:
    if 'fig' not in st.session_state:
        # Plotting
        st.header("Data Visualization")
        reduced_x = reduce_dimensions(X)
        reduced_data = pd.DataFrame(reduced_x, columns=['x', 'y'])
        reduced_data['label'] = st.session_state['model'].predict(X).astype(str)
        st.session_state.fig = px.scatter(reduced_data, x='x', y='y', color='label',
                                          color_discrete_sequence=px.colors.qualitative.Safe,
                                          title=f"the labels generated by {st.session_state['model']}")
        st.plotly_chart(st.session_state.fig)

        reduced_data['label'] = y.astype(str)
        st.session_state.fig2 = px.scatter(reduced_data, x='x', y='y', color='label',
                                           color_discrete_sequence=px.colors.qualitative.Safe, title=f"the actual labels")
        st.plotly_chart(st.session_state.fig2)

    else:
        st.write("Plotting")
        st.plotly_chart(st.session_state.fig)
        st.plotly_chart(st.session_state.fig2)
