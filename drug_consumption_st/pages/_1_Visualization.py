import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


import warnings
import tempfile


# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
sns.set_palette("Set2",2)

def criar_grafico():
    fig = sns.clustermap(df[['Nscore','Escore','Oscore','Ascore','Cscore', 'Coke']].corr(), figsize=(4, 4))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(temp_file.name)
    return temp_file.name


df = pd.read_csv("https://raw.githubusercontent.com/Andre647/Drug_Consumption/main/data/drug_consumption.csv", delimiter=',' )

df = df.drop(columns=[
    'ID','Gender','Ethnicity',
    'Country','Amphet',
    'Amyl','Caff',
    'Choc','Ketamine',
    'Legalh','Semer'
    ])
age_col = {
          -0.95197: '18-24',
          -0.07854: '25 - 34',
          0.49788: '35 - 44',
          1.09449: '45 - 54',
          1.82213: '55 - 64',
          2.59171: '65+'
          }
df['Age'] = df['Age'].replace(age_col)

education_col = {
            -2.43591: 'LSB16', # Left School Before 16...
            -1.73790: 'LS16',# Left School at 16
            -1.43719: 'LS17',# ...
            -1.22751: 'LS18',# ...
            -0.61113: 'Some College',
            -0.05921: 'Certificate',
            0.45468: 'University',
            1.16365: 'Masters',
            1.98437: 'Doctorate',
            }
df['Education'] = df['Education'].replace(education_col)

usage_col = {
    'CL0': 0,
    'CL1': 0,
    'CL2': 0,
    'CL3': 1,
    'CL4': 1,
    'CL5': 1,
    'CL6': 1
    }

# replacing the usage col
df_object = df[['Age','Education']]
df_numbers = df.select_dtypes('number')
df_drugs = df.select_dtypes('object').drop(columns=['Age','Education']).replace(usage_col)

# returning to the original state
df = df_object.join(df_numbers, how='outer')
df = df.join(df_drugs, how='outer')


"# EDA "
if st.checkbox('Show dataframe'): 
    st.dataframe(df, use_container_width=True)
    df.shape
st.divider()
"## Visualization"

plot_type = st.selectbox("Select plot", ["Drug Frequency", "Frequency per Age", "Frequency per Education"])

# Função para exibir o gráfico correspondente
def display_plot(plot_type):
    if plot_type == "Drug Frequency":
        st.write("## Drug Frequency")
        fig, axes = plt.subplots(4, 3, figsize=(10, 8), sharey=True)

        for ax, coluna in zip(axes.flatten(), df_drugs):
            sns.countplot(x=coluna, data=df, ax=ax)
            ax.set_ylabel('freq')


        plt.tight_layout()
        st.pyplot(fig)

    elif plot_type == "Frequency per Age":
        st.write("## Frequency per Age")
        fig, axes = plt.subplots(4, 3, figsize=(10, 8), sharey=True)

        for ax, coluna in zip(axes.flatten(), df_drugs):
            sns.countplot(x='Age', hue=coluna, data=df, ax=ax)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20)


        plt.tight_layout()
        st.pyplot(fig)

    elif plot_type == "Frequency per Education":
        st.write("## Frequency per Education")
        fig, axes = plt.subplots(4,3, figsize=(12, 8), sharey=True)

        for ax, coluna in zip(axes.flatten(), df_drugs):
            sns.countplot(x='Education', hue=coluna, data=df, ax=ax)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, fontsize=7)


        plt.tight_layout()
        st.pyplot(fig)

display_plot(plot_type)

"""### Interpretation

Something that needs to be pointed out is that the data comes from a survey, which means we cannot look at total values alone, as a specific group might have been interviewed more than another. For example, at first glance, it is easy to conclude from the graph that university students use more drugs, but is that really true?
"""

edu_df = pd.DataFrame(round(df.groupby('Education')['Coke'].sum() / df.groupby('Education')['Coke'].count(),2)*100).reset_index()
edu_df = edu_df.rename(columns={'Coke':'Coke Users (%)'})
edu_df['Total'] = df.groupby('Education')['Coke'].count().values

age_df = pd.DataFrame(round(df.groupby('Age')['Coke'].sum() / df.groupby('Age')['Coke'].count(),2)*100).reset_index()
age_df = age_df.rename(columns={'Coke':'Coke Users (%)'})
age_df['Total'] = df.groupby('Age')['Coke'].count().values

"""Well... It is!! As we can see, the highest concentration of cocaine users in the education sectors of our sample is among individuals with unfinished university degrees, however, it is worth noting that we have an unbalanced database with few samples; therefore, perhaps this analysis may not apply to the entire world. With more data, it would be possible to achieve greater accuracy in this information."""

left_column, right_column = st.columns(2)
left_column.dataframe(edu_df.style.highlight_max(axis=0))
right_column.dataframe(age_df.style.highlight_max(axis=0))

# Título da aplicação
st.divider()
st.title("Personality Test NEO-FFI-R")

# Definição das palavras e seus respectivos textos
conteudos = {
    "Nscore": "**Neuroticism**:  Individuals who score high are more likely to be moody and to experience such feelings as anxiety, worry, fear, anger, frustration, envy, jealousy, guilt, depressed mood, and loneliness.",
    "Escore": "**Extraversion**: It indicates how outgoing and social a person is. A person who scores high enjoy being with people, participating in social gatherings.",
    "Oscore": "**Openness to experience**: It indicates how open-minded a person is. A person with a high level enjoys trying new things. They are imaginative, curious, and open-minded. Individuals who are low in openness to experience would rather not try new things.",
    "Ascore": "**Agreeableness**: A person with a high level of agreeableness is usually warm, friendly, and tactful. They generally have an optimistic view of human nature and get along well with others.",
    "Cscore": "**Conscientiousness**: A person scoring high in conscientiousness usually has a high level of self-discipline. These individuals prefer to follow a plan, rather than act spontaneously. Their methodic planning and perseverance usually makes them highly successful in their chosen occupation."
}

grafico_path = criar_grafico()
left_column2, right_column2 = st.columns(2)

# Loop para criar um expander para cada palavra
for palavra, texto in conteudos.items():
    with st.expander(palavra):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(texto)
        with col2:
            st.image(grafico_path, use_column_width=True)


fig, axes = plt.subplots(1,3, figsize=(12, 6), sharey=True)

sns.histplot(data=df, x='Cscore', hue='Coke', ax=axes[0])
sns.histplot(data=df, x='Ascore', hue='Coke', ax=axes[1])
sns.histplot(data=df, x='Oscore', hue='Coke', ax=axes[2])

plt.subplots_adjust(wspace=0, hspace=0)
st.pyplot(fig)

"""As we can see, there is a slight tail to the left in the first two graphs, which could indicate that people with a **lack of discipline** and **pessimists** tend to use cocaine more frequently. Additionally, **open-minded** individuals have a slight tendency to use the drug.
"""
st.divider()
st.title("Personality Test BIS11 / ImpSS")

# Definição das palavras e seus respectivos textos
conteudos = {
    "Impulsive": "**Impulsiveness**: Tendency to act on a whim, displaying behavior characterized by little or no forethought, reflection, or consideration of the consequences.",
    "SS": "**Sensation**: Is input about the physical world obtained by our sensory receptors, and perception is the process by which the brain selects, organizes, and interprets these sensations. In other words, senses are the physiological basis of perception.",    
}

for palavra, texto in conteudos.items():
    with st.expander(palavra):
        st.write(texto)


fig, axes = plt.subplots(1,2, figsize=(10, 6), sharey=True)

sns.histplot(data=df, x='Impulsive', hue='Coke', ax=axes[0])
sns.histplot(data=df, x='SS', hue='Coke', ax=axes[1])

plt.subplots_adjust(wspace=0, hspace=0)
st.pyplot(fig)

"""These metrics are different from the others because they have a smaller number of possible values. However, it is notable that impulsive people and those who are perceptive of the world around them use the drug more than those who do not.

**With this, we conclude the visualizations of the variables in our problem. We have learned that, in our sample, university students aged between 18 and 24 have a higher likelihood of substance use. Additionally, pessimistic, impulsive, sensitive individuals, and those interested in experimenting new things show a greater tendency toward drug use.Those variables will be essential for our model.**
"""


