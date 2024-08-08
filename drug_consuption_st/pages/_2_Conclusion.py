
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from pages._1_Visualization import df


"# Classification "
# Separating the target variables from the independent variables
df_v = df[['Age','Education','Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']]
df_d = df[['Alcohol', 'Benzos', 'Cannabis', 'Coke', 'Crack','Ecstasy', 'Heroin', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']]
# hot encoding
hot_encode = OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first') #drop 'first' remove a multicolinearidade
hot_encode.fit(df_v.select_dtypes(exclude='number'))
df_v_hot = pd.DataFrame(hot_encode.transform(df_v.select_dtypes(exclude='number')),columns=hot_encode.get_feature_names_out())
df_v = df_v_hot.join(df_v.select_dtypes('number'))
# normalization
scaler = StandardScaler()
scaler.fit(df_v)
df_v_scaled = scaler.transform(df_v)
df_v_scaled = pd.DataFrame(df_v_scaled, columns=df_v.columns)
df_v = df_v_scaled

"""## Modeling

During this stage, I tested different models, such as Decision Trees, Logistic Regressions, Gaussian Naive Bayes, and MLPs. It appeared that Random Forests were the most effective, which is why this will be the model used. A good characteristic of Random Forest is the *balanced* parameter, which allows changing the class weights, making it suitable for dealing with imbalanced datasets. It's worth noting that, except for Nicotine and Cannabis, all our other targets are imbalanced.
"""

# Defining the model and the hyperparameters to be tested
base_estimator_un = RandomForestClassifier(class_weight='balanced', random_state=123) # For imbalanced classes
base_estimator = base_estimator = RandomForestClassifier(random_state=123)# For balanced classes
param_grid = {'n_estimators': [10,50,100,1000], 'criterion': ['gini', 'entropy', 'log_loss']}

# Applying Cross Validation to prevent overfitting and Grid Search to select hyperparameters
clf = GridSearchCV(base_estimator, param_grid, cv=5, scoring='f1')
clf_un = GridSearchCV(base_estimator_un, param_grid, cv=5, scoring='f1')

# Dictionary where the results will be stored
target_columns = df_d.columns
metrics = {coluna: {} for coluna in target_columns}

"### Training"

@st.cache_resource
def train_and_evaluate(df_v, df_d, _clf, _clf_un, metrics):

    models = {}

    # Função auxiliar para treinar e avaliar o modelo
    def process_target(col, classifier):
        y = df_d[col]
        X_train, X_test, y_train, y_test = train_test_split(df_v, y, stratify=y, test_size=0.3, random_state=123)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

  

        metrics[col] = {}
        metrics[col]['model'] = classifier.best_estimator_
        metrics[col]['report'] = classification_report(y_test, y_pred, output_dict=True)
        metrics[col]['confusion_matrix'] = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['Not User', 'User'])
        
        models[col] = classifier

    # Processar alvos específicos
    for col in df_d[['Nicotine', 'Cannabis']]:
        process_target(col, _clf)

    # Processar os demais alvos
    for col in df_d.drop(columns=['Cannabis', 'Nicotine']):
        process_target(col, _clf_un)

    return metrics, models

metrics_completed, models = train_and_evaluate(df_v, df_d, clf, clf_un, metrics)


"""## Eval"""

# transforming our metrics into a dataframe
results = pd.DataFrame(target_columns).rename(columns={0:'Drugs'})
results['accuracy'] = results['Drugs'].map(lambda x: metrics_completed[x]['report']['accuracy'])
for col in ['precision','recall','f1-score']:
    results[col] = results['Drugs'].map(
        lambda x: round(metrics_completed[x]['report']['0'][col],2)
    )
st.dataframe(results.style.highlight_max(axis=0))
st.divider()

###

plot_type = st.selectbox("Metric", ["f1-score","precision", "recall", "accuracy"])

# Função para exibir o gráfico correspondente
def display_plot(plot_type):
    if plot_type == "f1-score":
        fig,ax = plt.subplots(figsize=(10, 6))

        st.write("## F1-Score")
        sns.barplot(data=results.sort_values(by='f1-score', ascending=False), x='Drugs', y='f1-score', palette='rocket', ax=ax)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20, labelsize=10)


        plt.tight_layout()
        st.pyplot(fig)

    elif plot_type == "precision":
        fig, ax = plt.subplots(figsize=(10,6))

        st.write("## Precision(0)")
        sns.barplot(data=results.sort_values(by='precision', ascending=False), x='Drugs', y='precision', palette='rocket', ax=ax)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20, labelsize=10)


        plt.tight_layout()
        st.pyplot(fig)

    elif plot_type == "recall":
        fig, ax = plt.subplots(figsize=(10,6))

        st.write("## Recall(0)")
        sns.barplot(data=results.sort_values(by='recall', ascending=False), x='Drugs', y='recall', palette='rocket', ax=ax)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20, labelsize=10)


        plt.tight_layout()
        st.pyplot(fig)

    elif plot_type == "accuracy":
        fig, ax = plt.subplots(figsize=(10,6))

        st.write("## Accuracy")
        sns.barplot(data=results.sort_values(by='accuracy', ascending=False), x='Drugs', y='accuracy', palette='rocket', ax=ax)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20, labelsize=10)


        plt.tight_layout()
        st.pyplot(fig)   

display_plot(plot_type)



"""# Conclusion

## Results

**The model achieved above 70% F1-score for all drugs except Nicotine and Alcohol, maintaining an accuracy above 50% for all drugs except Alcohol.**

"""

st.subheader("**Considerations**:")
st.markdown("""We have imbalanced classes, where most of the respondents were not users of harder drugs, 
so it makes sense to have high scores for them. For example, in the dataframe, you can see 
that 96% of users did not use crack, so a simple deduction that all people are *non-users* 
of crack would give us an accuracy of 96%, which is not necessarily a positive thing. 
It would be like boasting about correctly identifying 96% of the teams that a group of people 
supports inside a stadium without mixed fans. The model has high precision because the metric 
does not consider false negatives, i.e., it does not calculate errors where the guess was a user, 
and the person in question was not;""")

st.markdown("However, in a problem like this, where an incentive to avoid drug use could be implemented, false positives are much more harmful, as they would fail to help someone who needs assistance. False negatives would only help someone who might not necessarily need it. Therefore, looking at the precision of the target 0 is a viable option in this case;")
st.markdown("With more data, we could have a more robust model, with more test sets and more diversification, which would ensure its applicability. However, the study proved satisfactory for what it proposes, showing that this type of analysis has potential.")

"""  

## What is the F1-score and why do we use it as a metric for model evaluation?

The F1 score is calculated as the harmonic mean of precision and recall. It provides a balance between precision and recall:


A high F1 score indicates both high precision and high recall, which implies that the model is performing well in terms of both identifying relevant instances and minimizing false positives. It is particularly useful in situations where you want to find an optimal balance between precision and recall, such as in binary classification tasks where the classes are imbalanced.

"""
