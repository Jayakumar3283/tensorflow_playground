import streamlit as st
st.set_page_config(
    page_title="tensorflow playground for classification task",
    page_icon="🧊",
)

st.sidebar.success("SELECT ONE OF THE PAGE")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras
from sklearn.datasets import make_classification, make_regression
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import InputLayer,Dense

st.title("Tenserflow playground Jayakumar Elite-13")
st.header("Random Data application")

classification_type = st.sidebar.selectbox('Specify the classification type',("Binary_class_classification","Multi_class_classification"))

if classification_type=="Binary_class_classification":
    ### number of samples
    num_samples = st.sidebar.slider("No_of samples", min_value=100, max_value=100000, value=500, step=1)

    ### random state or not
    random_state = st.sidebar.slider("Select Random state" ,min_value=1, max_value=100, value=0, step=1)

    ### No_of clusters per class
    n_clusters_per_class = st.sidebar.slider("No_of clusters per class", min_value=1, max_value=3, value=1, step=1)

    ### dist between classes
    class_sep = st.sidebar.slider("dist between class", min_value=1, max_value=3, value=2, step=1)

    #creation of dataset
    fv,cv=make_classification(n_samples=num_samples,
                              n_features=2,
                              n_informative=2,
                              n_redundant=0,
                              n_repeated=0,
                              n_classes=2,
                              n_clusters_per_class=n_clusters_per_class,
                              class_sep=class_sep,
                              random_state=random_state)
    df = pd.DataFrame(fv, columns=["Feature_1", "Feature_2"])
    df["Class_Label"] = cv

    ## creation of ANN model
    model = Sequential()
    model.add(InputLayer(input_shape=(2,)))

    hidden_layers = st.sidebar.number_input("No.of Hidden Layers", min_value=1, step=1)
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


    ### Learning rate selection
    learning_rate =  st.sidebar.slider("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001, help="define learning rate")
    sgd=SGD(learning_rate=learning_rate)      

    num_epochs = st.sidebar.slider("no.of epochs", min_value=1, max_value=50,step=1)
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=num_samples,step=1)

    if st.sidebar.button("Submit", type="primary"):
        ### Create a scatter plot
        st.write("Correlation between features wrt Class Label")

        fig = sns.scatterplot(data=df, x="Feature_1", y="Feature_2", hue="Class_Label", palette="viridis")
        fig.set_xlabel('Feature_1')
        fig.set_ylabel('Feature_2')
        st.pyplot(fig.figure)

        y=df['Class_Label']
        X=df[['Feature_1','Feature_2']]
    
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(fv,cv,test_size=0.3,stratify=cv)

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)

        model.summary()
        model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

        history=model.fit(X_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_split=0.3,steps_per_epoch=700//batch_size)
        # Streamlit app
        st.title("Training and Validation Loss Plot")
        st.write("### Plotting Loss Curves")

        # Create plot
        fig,ax = plt.subplots()
        plt.plot(range(1, num_epochs+1), history.history['loss'], label='Train')
        plt.plot(range(1, num_epochs+1), history.history['val_loss'], label='Test')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        st.pyplot(fig)

        from mlxtend.plotting import plot_decision_regions

        # Plot decision regions
        fig, ax = plt.subplots()
        plot_decision_regions(X_test, y_test, clf=model, ax=ax)
        ax.set_xlabel('Feature_1')
        ax.set_ylabel('Feature_2')
        ax.set_title('Decision Regions')
        st.pyplot(fig)
elif classification_type == "Multi_class_classification":

    ### number of samples
    num_samples = st.sidebar.slider("No_of samples", min_value=100, max_value=100000, value=500, step=1)

    ### random state or not
    random_state = st.sidebar.slider("Select Random state" ,min_value=1, max_value=100, value=0, step=1)

    ### No_of clusters per class
    n_clusters_per_class = st.sidebar.slider("No_of clusters per class", min_value=1, max_value=3, value=1, step=1)

    ### dist between classes
    class_sep = st.sidebar.slider("dist between class", min_value=1, max_value=3, value=2, step=1)

    ### no.of classes
    n_classses = st.sidebar.slider("no.of classes", min_value=3, max_value=5, value=3, step=1)

    #creation of dataset
    fv,cv=make_classification(n_samples=num_samples,
                              n_features=2,
                              n_informative=2,
                              n_redundant=0,
                              n_repeated=0,
                              n_classes=n_classses,
                              n_clusters_per_class=n_clusters_per_class,
                              class_sep=class_sep,
                              random_state=random_state)
    df = pd.DataFrame(fv, columns=["Feature_1", "Feature_2"])
    df["Class_Label"] = cv

    ## creation of ANN model
    model = Sequential()
    model.add(InputLayer(input_shape=(2,)))

    hidden_layers = st.sidebar.number_input("No.of Hidden Layers", min_value=1, step=1)
    hidden_layers_configuration = []

    for layer in range(1, hidden_layers + 1):
        if hidden_layers != layer:
            neurons = st.sidebar.number_input(f"Number of Neurons in Layer {layer}", min_value=1, step=1)
            activation = st.sidebar.selectbox(f"Activation Function for Layer {layer}", ["relu", "sigmoid", "tanh"])
            hidden_layers_configuration.append((neurons, activation))
        else:
            neurons = st.sidebar.number_input(f"Number of Neurons in Layer {layer}", min_value=n_classses, max_value=n_classses, step=1)
            activation = st.sidebar.selectbox(f"Activation Function for Layer {layer}", ["softmax"])
            hidden_layers_configuration.append((neurons, activation))

    for neurons, activation in hidden_layers_configuration:
        model.add(Dense(neurons, activation=activation, use_bias=True))


    ### Learning rate selection
    learning_rate =  st.sidebar.slider("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001, help="define learning rate")
    sgd=SGD(learning_rate=learning_rate)      

    num_epochs = st.sidebar.slider("no.of epochs", min_value=1, max_value=50,step=1)
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=num_samples,step=1)

    if st.sidebar.button("Submit", type="primary"):
        ### Create a scatter plot
        st.write("Correlation between features wrt Class Label")

        fig = sns.scatterplot(data=df, x="Feature_1", y="Feature_2", hue="Class_Label", palette="viridis")
        fig.set_xlabel('Feature_1')
        fig.set_ylabel('Feature_2')
        st.pyplot(fig.figure)

        y=df['Class_Label']
        X=df[['Feature_1','Feature_2']]
    
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(fv,cv,test_size=0.3,stratify=cv)

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)

        model.summary()
        model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        history=model.fit(X_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_split=0.3,steps_per_epoch=700//batch_size)
        # Streamlit app
        st.title("Training and Validation Loss Plot")
        st.write("### Plotting Loss Curves")

        # Create plot
        fig,ax = plt.subplots()
        plt.plot(range(1, num_epochs+1), history.history['loss'], label='Train')
        plt.plot(range(1, num_epochs+1), history.history['val_loss'], label='Test')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        st.pyplot(fig)

        
    