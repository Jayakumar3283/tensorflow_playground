import streamlit as st
st.set_page_config(
    page_title="tensorflow playground for unique data sets",
    page_icon="🧊",
)
st.title("Tenserflow playground Jayakumar Elite-13")

import pandas as pd
import numpy as np ##"C:\Users\JAY KUMAR\deep_learning\Tasks\ANN_MODEL\1.ushape.csv"
df1=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\1.ushape.csv',names=["Feature 1", "Feature 2","Class Label"])
df2=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\2.concerticcir1.csv',names=["Feature 1", "Feature 2","Class Label"])
df3=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\3.concertriccir2.csv',names=["Feature 1", "Feature 2","Class Label"])
df4=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\4.linearsep.csv',names=["Feature 1", "Feature 2","Class Label"])
df5=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\5.outlier.csv',names=["Feature 1", "Feature 2","Class Label"])
df6=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\6.overlap.csv',names=["Feature 1", "Feature 2","Class Label"])
df7=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\7.xor.csv',names=["Feature 1", "Feature 2","Class Label"])
df8=pd.read_csv('C:\\Users\\JAY KUMAR\\deep_learning\\Tasks\\ANN_MODEL\\8.twospirals.csv',names=["Feature 1", "Feature 2","Class Label"])

dataframes = {'1.ushape': df1, '2.concerticcir1': df2,
              '3.concerticcir2':df3,'4.linearsep':df4,
              '5.outlier.csv':df5,'6.overlap':df6,
              '7.xor':df7,'8.twospirals':df8}

selected_df = st.sidebar.selectbox('Select Dataframe', list(dataframes.keys()))
data=dataframes[selected_df]
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import keras

from sklearn.datasets import make_classification,make_regression
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import InputLayer,Dense




#num_samples
num_samples = st.sidebar.slider("No_of samples", min_value=1, max_value=1000, value=10, step=1)

#random state or not

random_sates = st.sidebar.slider("Select Random state" ,min_value=1, max_value=100, value=0, step=1)


# Learning rate selection
learning_rates =  st.sidebar.slider("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001,
                                    help="Adjust the learning rate for the model.")

sgd=SGD(learning_rate=learning_rates)

model = Sequential()
model.add(InputLayer(input_shape=(2,)))

hidden_layers = st.sidebar.number_input("Hidden Layers", min_value=1, step=1)
hidden_layers_configuration = []
for layer in range(1, hidden_layers + 1):
    if hidden_layers != layer:
        neurons = st.sidebar.number_input(f"Number of Neurons in Layer {layer}", min_value=1, step=1)
        activation = st.sidebar.selectbox(f"Activation Function for Layer {layer}", ["relu", "sigmoid", "tanh","softmax"])
        hidden_layers_configuration.append((neurons, activation))
    else:
        neurons = st.sidebar.number_input(f"Number of Neurons in Layer {layer}", min_value=1, max_value=1, step=1)
        activation = st.sidebar.selectbox(f"Activation Function for Layer {layer}", ["sigmoid"])
        hidden_layers_configuration.append((neurons, activation))

for neurons, activation in hidden_layers_configuration:
    model.add(Dense(neurons, activation=activation, use_bias=True))

num_epochs = st.sidebar.slider("no.of epochs", min_value=1, max_value=50,step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=num_samples,step=1)


sample_data=data.sample(n=num_samples,random_state=random_sates,replace=True)

if st.sidebar.button("Submit", type="primary"):

    st.header(f"Selected Data application : {selected_df}")


    sample_data['Class Label']=sample_data['Class Label'].astype("int64")

    # Create scatter plot
    st.write("### Correlation between features wrt Class Label")

    fig = sns.scatterplot(data=sample_data, x="Feature 1", y="Feature 2", hue="Class Label", palette="viridis")
    fig.set_xlabel('Feature 1')
    fig.set_ylabel('Feature 2')
    st.pyplot(fig.figure)
    y=sample_data['Class Label']
    X=sample_data[['Feature 1','Feature 2']]


    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_tset = train_test_split(X,y,test_size=0.3)

    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    std=StandardScaler()
    X_train=std.fit_transform(X_train)
    X_test=std.transform(X_test)

    st.write(X_test.dtype,y_tset.dtype)


    model.summary()
    model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

    history=model.fit(X_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_split=0.3,steps_per_epoch=700//20)
    history
    # Streamlit app
    st.title("Training and Validation Loss Plot")
    st.write("### Plotting Loss Curves")

    # Create plot
    fig,ax = plt.subplots()
    plt.plot(range(1, num_epochs+1), history.history['loss'], label='Train')
    plt.plot(range(1, num_epochs+1), history.history['val_loss'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    st.pyplot(fig)

    from mlxtend.plotting import plot_decision_regions

    # Plot decision regions
    fig, ax = plt.subplots()
    plot_decision_regions(X_test, y_tset.values,clf=model)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Regions')
    st.pyplot(fig)