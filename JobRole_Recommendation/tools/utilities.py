import streamlit as st
import pandas as pd
import base64
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import os

pl_list = ['Python', 'R', 'SQL', 'C' , 'C++', 'Java', 'Javascript', 'Julia', 'Bash', 'MATLAB', 'Other Lang']
dv_tools_list = ['Matplotlib', 'Seaborn','Ploty','Ggplot','Shiny','D3','Altair','Bokeh','Geoplotlib','Leaflet','Other Vis']

def load_css():

    with open("tools/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)

def generate_download_link(csv_data, filename):
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download the file with recommendation</a>'
    return href

    
@st.cache_data    
def load_data(file):
    print("Current Working Directory inside load_data function:", os.getcwd())
    data = pd.read_csv(file)
    data = pd.DataFrame(data)

    return data

def fetch_processed_data_for_timetrend():

    k20 = load_data("data/kaggle_survey_2020_responses.csv")
    k21 = load_data("data/kaggle_survey_2021_responses.csv")
    k22 = load_data("data/kaggle_survey_2022_responses.csv")

    #filter necessary columns, create new column to indicate year for each dataframe
    new20 = k20[['Q1', 'Q3', 'Q6','Q37_Part_1','Q37_Part_2','Q37_Part_3','Q37_Part_4','Q37_Part_5','Q37_Part_6','Q37_Part_7','Q37_Part_8','Q37_Part_9', 'Q37_Part_10','Q37_Part_11','Q37_OTHER']]
    new20.insert(0, "Survey Year", 2020)
    new21 = k21[['Q1', 'Q3', 'Q6','Q40_Part_1','Q40_Part_2','Q40_Part_3','Q40_Part_4','Q40_Part_5','Q40_Part_6','Q40_Part_7','Q40_Part_8','Q40_Part_9', 'Q40_Part_10','Q40_Part_11','Q40_OTHER']]
    new21.insert(0, "Survey Year", 2021)
    new22 = k22[['Q2', 'Q4', 'Q11','Q6_1','Q6_2','Q6_3','Q6_4','Q6_5','Q6_6','Q6_7','Q6_8','Q6_9','Q6_10', 'Q6_11','Q6_12']]
    new22.insert(0, "Survey Year", 2022)

    #rename the columns
    column_names = ['Year','Age','Country','Experience','Coursera','edX','Kaggle Learn Courses','DataCamp','Fast.ai','Udacity','Udemy','LinkedIn Learning','Cloud-Cert programs','University Courses','None','Other']
    new20.columns=column_names
    new21.columns=column_names
    new22.columns=column_names

    #concat the 3 dataframes together
    data = pd.concat([new20,new21,new22])
    data = data.drop(data.index[0])

    #group certain age categories together
    data['Age'] = data['Age'].str.replace("18-21","18-24")
    data['Age'] = data['Age'].str.replace("22-24","18-24")
    #group certain age categories together
    data['Experience'] = data['Experience'].str.replace("1-2 years","1-3 years")
    data = data.reset_index(drop=True)

    for index, row in data.iterrows():
        if (row['University Courses'] == 'University Courses (resulting in a university degree)'):
            data.at[index,'University Courses'] = 'University Courses'
        if(row['Cloud-Cert programs'] == 'Cloud-certification programs (direct from AWS, Azure, GCP, or similar)'):
            data.at[index,'Cloud-Cert programs'] = 'Cloud-Cert programs'

    return data;

def fetch_processed_exploratory_data():

    k20 = load_data("data/kaggle_survey_2020_responses.csv")
    k21 = load_data("data/kaggle_survey_2021_responses.csv")
    k22 = load_data("data/kaggle_survey_2022_responses.csv")

    #filter necessary columns (age, experience,yearly salary and education), create new column to indicate year for each dataframe
    new20 = k20[['Q1','Q6','Q4','Q5','Q24']]
    new20.insert(0, "Year", 2020)
    new21 = k21[['Q1','Q6','Q4','Q5','Q25']]
    new21.insert(0, "Year", 2021)
    new22 = k22[['Q2','Q11','Q8','Q23','Q29']]
    new22.insert(0, "Year", 2022)

    #rename columns of all dataframes
    new20.columns = list(['Year','Age','Experience','Education','Job Role', 'Yearly Salary'])
    new21.columns = list(['Year','Age','Experience','Education','Job Role', 'Yearly Salary'])
    new22.columns = list(['Year','Age','Experience','Education','Job Role', 'Yearly Salary'])

    #Salary range is inconsistent across dataframes
    for index, row in new22.iterrows():
        c = row['Yearly Salary']
        if c == '$500,000-999,999' or c == '>$1,000,000':
            new22.at[index, 'Yearly Salary'] = '> $500,000'

    for index, row in new21.iterrows():
        c = row['Yearly Salary']
        if c == '$500,000-999,999' or c == '>$1,000,000':
            new21.at[index, 'Yearly Salary'] = '> $500,000'

    #join all 3 dataframes together, add column names
    data = pd.concat([new20,new21,new22])
    #fill NA with "noreply"
    data = data.fillna("Not Answered")

    #group certain age categories together
    data['Age'] = data['Age'].str.replace("18-21","18-24")
    data['Age'] = data['Age'].str.replace("22-24","18-24")

    #group certain age categories together
    data['Experience'] = data['Experience'].str.replace("1-2 years","1-3 years")

    #group certain role categories together
    data['Job Role'] = data['Job Role'].str.replace("Product/Project Manager","Manager")
    data['Job Role'] = data['Job Role'].str.replace("Program/Project Manager","Manager")
    data['Job Role'] = data['Job Role'].str.replace("Product Manager","Manager")
    data['Job Role'] = data['Job Role'].str.replace("Manager (Program, Project, Operations, Executive-level, etc)","Manager", regex=False)
    data['Job Role'] = data['Job Role'].str.replace("Machine Learning Engineer","ML Engineer")
    data['Job Role'] = data['Job Role'].str.replace("Machine Learning/ MLops Engineer","ML Engineer")
    data['Job Role'] = data['Job Role'].str.replace("Data Analyst (Business, Marketing, Financial, Quantitative, etc)","Data Analyst", regex=False)
    data['Job Role'] = data['Job Role'].str.replace("Developer Advocate","Developer Relations/Advocacy")
    
    data=data.drop(data.index[0])

    return data

def fetch_recommender_data():
    
    k20 = load_data("data/kaggle_survey_2020_responses.csv")
    k21 = load_data("data/kaggle_survey_2021_responses.csv")
    k22 = load_data("data/kaggle_survey_2022_responses.csv")

    # filter necessary columns, create new column to indicate year for each dataframe
    new20 = k20[['Q1', 'Q2', 'Q24', 'Q3', 'Q6', 'Q4', 'Q5', 'Q7_Part_1', 'Q7_Part_2', 'Q7_Part_3', 'Q7_Part_4', 'Q7_Part_5',
                'Q7_Part_6', 'Q7_Part_7', 'Q7_Part_8', 'Q7_Part_9', 'Q7_Part_10', 'Q7_Part_11', 'Q7_Part_12', 'Q7_OTHER',
                'Q14_Part_1', 'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6', 'Q14_Part_7',
                'Q14_Part_8', 'Q14_Part_9', 'Q14_Part_10', 'Q14_OTHER', 'Q15', 'Q22', 'Q25']]
    new20.insert(0, "Survey Year", 2020)
    new21 = k21[['Q1', 'Q2', 'Q25', 'Q3', 'Q6', 'Q4', 'Q5', 'Q7_Part_1', 'Q7_Part_2', 'Q7_Part_3', 'Q7_Part_4', 'Q7_Part_5',
                'Q7_Part_6', 'Q7_Part_7', 'Q7_Part_8', 'Q7_Part_9', 'Q7_Part_10', 'Q7_Part_11', 'Q7_Part_12', 'Q7_OTHER',
                'Q14_Part_1', 'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6', 'Q14_Part_7',
                'Q14_Part_8', 'Q14_Part_9', 'Q14_Part_10', 'Q14_OTHER', 'Q15', 'Q23', 'Q26']]
    new21.insert(0, "Survey Year", 2021)
    new22 = k22[['Q2', 'Q3', 'Q29', 'Q4', 'Q11', 'Q8', 'Q23', 'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5', 'Q12_6', 'Q12_7',
                'Q12_8', 'Q12_9', 'Q12_10', 'Q12_11', 'Q12_12', 'Q12_13', 'Q12_14', 'Q12_15', 'Q15_1', 'Q15_2', 'Q15_3',
                'Q15_4', 'Q15_5', 'Q15_6', 'Q15_7', 'Q15_8', 'Q15_9', 'Q15_10', 'Q15_15', 'Q16', 'Q27', 'Q30']]
    new22.insert(0, "Survey Year", 2022)

    # 2020 and 2021 does not include C# and GO options, while 2022 does not include Swift option,
    # sub these options into the “Others” category
    current = 0
    for row in new22.itertuples():
        a = row.Q12_5
        b = row.Q12_13
        if a == 'C#' or b == 'Go':
            new22.loc[current, 'Q12_15'] = 'Other'
        current += 1
    current = 0
    for row in new21.itertuples():
        a = row.Q7_Part_9
        if a == 'Swift':
            new21.loc[current, 'Q7_OTHER'] = 'Other'
        current += 1
    current = 0
    for row in new20.itertuples():
        a = row.Q7_Part_9
        if a == 'Swift':
            new20.loc[current, 'Q7_OTHER'] = 'Other'
        current += 1

    # remove certain columns
    new20 = new20[
        ['Survey Year', 'Q1', 'Q2', 'Q24', 'Q3', 'Q6', 'Q4', 'Q5', 'Q7_Part_1', 'Q7_Part_2', 'Q7_Part_3', 'Q7_Part_4',
        'Q7_Part_5', 'Q7_Part_6', 'Q7_Part_7', 'Q7_Part_8', 'Q7_Part_10', 'Q7_Part_11', 'Q7_OTHER', 'Q14_Part_1',
        'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6', 'Q14_Part_7', 'Q14_Part_8', 'Q14_Part_9',
        'Q14_Part_10', 'Q14_OTHER', 'Q15', 'Q22', 'Q25']]
    new21 = new21[
        ['Survey Year', 'Q1', 'Q2', 'Q25', 'Q3', 'Q6', 'Q4', 'Q5', 'Q7_Part_1', 'Q7_Part_2', 'Q7_Part_3', 'Q7_Part_4',
        'Q7_Part_5', 'Q7_Part_6', 'Q7_Part_7', 'Q7_Part_8', 'Q7_Part_10', 'Q7_Part_11', 'Q7_OTHER', 'Q14_Part_1',
        'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6', 'Q14_Part_7', 'Q14_Part_8', 'Q14_Part_9',
        'Q14_Part_10', 'Q14_OTHER', 'Q15', 'Q23', 'Q26']]
    new22 = new22[
        ['Survey Year', 'Q2', 'Q3', 'Q29', 'Q4', 'Q11', 'Q8', 'Q23', 'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_6', 'Q12_7',
        'Q12_8', 'Q12_12', 'Q12_9', 'Q12_11', 'Q12_15', 'Q15_1', 'Q15_2', 'Q15_3', 'Q15_4', 'Q15_5', 'Q15_6', 'Q15_7',
        'Q15_8', 'Q15_9', 'Q15_10', 'Q15_15', 'Q16', 'Q27', 'Q30']]

    # rename 2022 columns so that all columns names are the same
    new20 = new20.rename(
        columns={'Survey Year': 'Year', 'Q1': 'Age', 'Q2': 'Gender', 'Q24': 'Income', 'Q3': 'Country', 'Q6': 'Experience',
                'Q4': 'Education', 'Q5': 'Job Role', 'Q7_Part_1': 'Python', 'Q7_Part_2': 'R', 'Q7_Part_3': 'SQL',
                'Q7_Part_4': 'C', 'Q7_Part_5': 'C++', 'Q7_Part_6': 'Java', 'Q7_Part_7': 'Javascript', 'Q7_Part_8': 'Julia',
                'Q7_Part_10': 'Bash', 'Q7_Part_11': 'MATLAB', 'Q7_OTHER': 'Other Lang', 'Q14_Part_1': 'Matplotlib',
                'Q14_Part_2': 'Seaborn', 'Q14_Part_3': 'Ploty', 'Q14_Part_4': 'Ggplot', 'Q14_Part_5': 'Shiny',
                'Q14_Part_6': 'D3', 'Q14_Part_7': 'Altair', 'Q14_Part_8': 'Bokeh', 'Q14_Part_9': 'Geoplotlib',
                'Q14_Part_10': 'Leaflet', 'Q14_OTHER': 'Other Vis', 'Q15': 'ML Years', 'Q22': 'ML in Job',
                'Q25': 'Spending on ML'})
    new21 = new21.rename(
        columns={'Survey Year': 'Year', 'Q1': 'Age', 'Q2': 'Gender', 'Q25': 'Income', 'Q3': 'Country', 'Q6': 'Experience',
                'Q4': 'Education', 'Q5': 'Job Role', 'Q7_Part_1': 'Python', 'Q7_Part_2': 'R', 'Q7_Part_3': 'SQL',
                'Q7_Part_4': 'C', 'Q7_Part_5': 'C++', 'Q7_Part_6': 'Java', 'Q7_Part_7': 'Javascript', 'Q7_Part_8': 'Julia',
                'Q7_Part_10': 'Bash', 'Q7_Part_11': 'MATLAB', 'Q7_OTHER': 'Other Lang', 'Q14_Part_1': 'Matplotlib',
                'Q14_Part_2': 'Seaborn', 'Q14_Part_3': 'Ploty', 'Q14_Part_4': 'Ggplot', 'Q14_Part_5': 'Shiny',
                'Q14_Part_6': 'D3', 'Q14_Part_7': 'Altair', 'Q14_Part_8': 'Bokeh', 'Q14_Part_9': 'Geoplotlib',
                'Q14_Part_10': 'Leaflet', 'Q14_OTHER': 'Other Vis', 'Q15': 'ML Years', 'Q23': 'ML in Job',
                'Q26': 'Spending on ML'})
    new22 = new22.rename(
        columns={'Survey Year': 'Year', 'Q2': 'Age', 'Q3': 'Gender', 'Q29': 'Income', 'Q4': 'Country', 'Q11': 'Experience',
                'Q8': 'Education', 'Q23': 'Job Role', 'Q12_1': 'Python', 'Q12_2': 'R', 'Q12_3': 'SQL', 'Q12_4': 'C',
                'Q12_6': 'C++', 'Q12_7': 'Java', 'Q12_8': 'Javascript', 'Q12_12': 'Julia', 'Q12_9': 'Bash',
                'Q12_11': 'MATLAB', 'Q12_15': 'Other Lang', 'Q15_1': 'Matplotlib', 'Q15_2': 'Seaborn', 'Q15_3': 'Ploty',
                'Q15_4': 'Ggplot', 'Q15_5': 'Shiny', 'Q15_6': 'D3', 'Q15_7': 'Altair', 'Q15_8': 'Bokeh',
                'Q15_9': 'Geoplotlib', 'Q15_10': 'Leaflet', 'Q15_15': 'Other Vis', 'Q16': 'ML Years', 'Q27': 'ML in Job',
                'Q30': 'Spending on ML'})

    # concat the 3 dataframes together

    data2 = pd.concat([new20, new21])
    data = pd.concat([data2, new22])

    data = data.tail(-1)
    data.reset_index(drop=True, inplace=True)

    print(data.shape)

    # group certain age categories together
    data['Age'] = data['Age'].str.replace("18-21", "18-24")
    data['Age'] = data['Age'].str.replace("22-24", "18-24")
    # group certain age categories together
    data['Experience'] = data['Experience'].str.replace("1-2 years", "1-3 years")
    # group certain role categories together
    data['Job Role'] = data['Job Role'].str.replace("Product/Project Manager", "Manager")
    data['Job Role'] = data['Job Role'].str.replace("Program/Project Manager", "Manager")
    data['Job Role'] = data['Job Role'].str.replace("Product Manager", "Manager")
    data['Job Role'] = data['Job Role'].str.replace("Manager (Program, Project, Operations, Executive-level, etc)",
                                                    "Manager", regex=False)
    data['Job Role'] = data['Job Role'].str.replace("Machine Learning Engineer", "ML Engineer")
    data['Job Role'] = data['Job Role'].str.replace("Machine Learning/ MLops Engineer", "ML Engineer")
    data['Job Role'] = data['Job Role'].str.replace("Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
                                                    "Data Analyst", regex=False)
    data['Job Role'] = data['Job Role'].str.replace("Developer Advocate", "Developer Relations/Advocacy")
    data['Income'] = data['Income'].str.replace("$500,000-999,999", "> $500,000", regex=False)
    data['Income'] = data['Income'].str.replace(">$1,000,000", "> $500,000", regex=False)
    data['Income'] = data['Income'].str.replace("300,000-499,999", "300,000-500,000", regex=False)

    # replacing other minority gender categories as "Others"
    data['Gender'] = data['Gender'].replace(['Prefer not to say', 'Nonbinary', 'Prefer to self-describe'], 'Others')

    data = data.drop(data[data['Job Role'] == 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].index)

    #dropping data points with irrelevant job roles
    data = data.dropna(subset=['Job Role'])
    data = data.drop(data[data['Job Role']=='Student'].index)
    data = data.drop(data[data['Job Role']=='Currently not employed'].index)
    data = data.drop(data[data['Job Role']=='Other'].index)

    return data;

# Encoding the categorical variables into numerical variables
class IncomeTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        proportions = df['Income'].value_counts(normalize=True)
        unique_income_ranges = proportions.index
        df['Income'] = df['Income'].fillna(
            pd.Series(np.random.choice(unique_income_ranges, p=proportions, size=len(df))))

        OHE_Income = []

        for index, row in df.iterrows():
            row_income_str = row['Income']
            if (row_income_str == "$0-999"):
                OHE_Income.append(500)
            elif (row_income_str == "> $500,000"):
                OHE_Income.append(500000)
            else:
                row_income_str = row_income_str.replace(",", "")
                row_income_str = row_income_str.strip()
                lower, upper = row_income_str.split("-")
                mean_income = (float(lower) + float(upper)) / 2
                OHE_Income.append(mean_income)

        OHE_Income = np.array(OHE_Income)

        min_val = np.min(OHE_Income)
        max_val = np.max(OHE_Income)
        print("Min Income ", min_val)
        print("Max Income ", max_val)

        OHE_Income = (OHE_Income - min_val) / (max_val - min_val)

        OHE_Income = OHE_Income.reshape(-1, 1)

        return OHE_Income


# Binning the experience to numerical mean in the range
class ExpTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        # print(df['Experience'].value_counts(dropna=False))

        proportions = df['Experience'].value_counts(normalize=True)
        unique_age_ranges = proportions.index
        df['Experience'] = df['Experience'].fillna(
            pd.Series(np.random.choice(unique_age_ranges, p=proportions, size=len(df))))

        # print(df['Experience'].value_counts(dropna=False))

        OHE_Exp = []

        for index, row in df.iterrows():
            row_exp_str = row['Experience']
            if (row_exp_str == "I have never written code"):
                OHE_Exp.append(0)
            elif (row_exp_str == "< 1 years"):
                OHE_Exp.append(0.5)
            elif (row_exp_str == "20+ years"):
                OHE_Exp.append(20)
            else:
                row_exp_str = row_exp_str.replace("years", "")
                row_exp_str = row_exp_str.strip()
                lower, upper = row_exp_str.split("-")
                mean_exp = (float(lower) + float(upper)) / 2
                OHE_Exp.append(mean_exp)

        OHE_Exp = np.array(OHE_Exp)
        OHE_Exp = OHE_Exp.reshape(-1, 1)

        print("Shape of the OHE_Exp: ", OHE_Exp.shape)

        return OHE_Exp


# Binning the age values to numerical mean in the range
class AgeTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        OHE_Age = []

        for index, row in df.iterrows():
            row_age_str = row['Age']
            if (row_age_str == "70+"):
                OHE_Age.append(75)
            else:
                lower, upper = row_age_str.split("-")
                mean_age = (float(lower) + float(upper)) / 2
                OHE_Age.append(mean_age)

        OHE_Age = np.array(OHE_Age)
        min_val = np.min(OHE_Age)
        max_val = np.max(OHE_Age)
        OHE_Age = (OHE_Age - min_val) / (max_val - min_val)
        OHE_Age = OHE_Age.reshape(-1, 1)

        print("Shape of the OHE_Age: ", OHE_Age.shape)

        return OHE_Age

# Ordinal encoding for education column in data
class EduTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        OHE_Edu = []

        for index, row in df.iterrows():
            if (df.at[index, 'Education'] == "I prefer not to answer"):
                OHE_Edu.append(0)
            elif (df.at[index, 'Education'] == "No formal education past high school"):
                OHE_Edu.append(1)
            elif (df.at[index, 'Education'] == "Some college/university study without earning a bachelor’s degree"):
                OHE_Edu.append(2)
            elif (df.at[index, 'Education'] == "Bachelor’s degree"):
                OHE_Edu.append(3)
            elif (df.at[index, 'Education'] == "Master’s degree"):
                OHE_Edu.append(4)
            elif (df.at[index, 'Education'] == "Professional degree"):
                OHE_Edu.append(4)
            elif (df.at[index, 'Education'] == "Doctoral degree"):
                OHE_Edu.append(5)
            elif (df.at[index, 'Education'] == "Professional doctorate"):
                OHE_Edu.append(6)

        OHE_Edu = np.array(OHE_Edu)
        OHE_Edu = OHE_Edu.reshape(-1, 1)

        print("Shape of the OHE_Edu: ", OHE_Edu.shape)

        return OHE_Edu


# Encoding the Known_PL

class PLTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        OHE_PL = []

        for index, row in df.iterrows():
            row_PL = []
            for item in pl_list:
                if (pd.isna(df.at[index, item])):
                    row_PL.append(0)
                else:
                    row_PL.append(1)
            OHE_PL.append(row_PL)

        OHE_PL = np.array(OHE_PL)

        print("Shape of the OHE_PL: ", OHE_PL.shape)

        return OHE_PL

# Encoding the countries
class CountryTransformer:
    def fit(self, df, y=None, **fit_params):
        return df

    def transform(self, df, y=None, **fit_params):
        nrows = len(df)
        # Sorted option to form fields
        country_list_all_data = fetch_recommender_data()['Country']
        unique_country_list = country_list_all_data.unique()
        ncolumns = len(unique_country_list)

        df_Country = pd.DataFrame(0, index=range(nrows), columns=unique_country_list)
        OHE_Country = []

        df = df.reset_index(drop=True)

        for index, row in df.iterrows():
            row_Country = row['Country']
            df_Country.at[index, row_Country] = 1

        OHE_Country = df_Country.to_numpy()

        print("Shape of the OHE_Country: ", OHE_Country.shape)

        return OHE_Country


# Encoding the Data Visualisation tools

class DataVisTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        OHE_DV = []

        for index, row in df.iterrows():
            row_DV_count = 0
            for item in dv_tools_list:
                if not (pd.isna(df.at[index, item])):
                    row_DV_count = row_DV_count + 1

            OHE_DV.append(row_DV_count)

        OHE_DV = np.array(OHE_DV)
        OHE_DV = OHE_DV.reshape(-1, 1)

        print("Shape of the OHE_DV: ", OHE_DV.shape)

        return OHE_DV


# Encoding the ML Years

class MLYearsTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        # print(df['ML Years'].value_counts(dropna=False))

        proportions = df['ML Years'].value_counts(normalize=True)
        unique_ml_year_ranges = proportions.index
        df['ML Years'] = df['ML Years'].fillna(
            pd.Series(np.random.choice(unique_ml_year_ranges, p=proportions, size=len(df))))

        # print(unique_ml_year_ranges)
        # print(df['ML Years'].value_counts(dropna=False))

        OHE_MLY = []

        for index, row in df.iterrows():
            row_MLY_str = row['ML Years']
            if (row_MLY_str == "I do not use machine learning methods"):
                OHE_MLY.append(0)
            elif (row_MLY_str == "Under 1 year"):
                OHE_MLY.append(0.5)
            elif (row_MLY_str == "20 or more years"):
                OHE_MLY.append(20)
            else:
                # print(row_MLY_str)
                row_MLY_str = row_MLY_str.replace("years", "")
                row_MLY_str = row_MLY_str.strip()
                lower, upper = row_MLY_str.split("-")
                mean_exp = (float(lower) + float(upper)) / 2
                OHE_MLY.append(mean_exp)

        OHE_MLY = np.array(OHE_MLY)

        min_val = np.min(OHE_MLY)
        max_val = np.max(OHE_MLY)
        OHE_MLY = (OHE_MLY - min_val) / (max_val - min_val)

        print(min_val)
        print(max_val)

        OHE_MLY = OHE_MLY.reshape(-1, 1)

        print("Shape of the OHE_MLY: ", OHE_MLY.shape)

        return OHE_MLY

# Encoding Gender
class GenderTransformer:
    def fit(self, df, y=None, **fit_params):
        return df

    def transform(self, df, y=None, **fit_params):
        nrows = len(df)

        gender_list = fetch_recommender_data()['Gender'].unique()
        df_Gender = pd.DataFrame(0, index=range(nrows), columns=gender_list)
        OHE_Gender = []

        df = df.reset_index(drop=True)

        for index, row in df.iterrows():
            row_Gender = row['Gender']
            df_Gender.at[index, row_Gender] = 1

        OHE_Gender = df_Gender.to_numpy()

        print("Shape of the OHE_Country: ", OHE_Gender.shape)

        return OHE_Gender


# Ordinal encoding for ML incoporation in job
class MLinJobTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        proportions = df['ML in Job'].value_counts(normalize=True)
        unique_mlinjob_ranges = proportions.index
        df['ML in Job'] = df['ML in Job'].fillna(
            pd.Series(np.random.choice(unique_mlinjob_ranges, p=proportions, size=len(df))))

        OHE_MLinJob = []

        for index, row in df.iterrows():
            row_MLinJob = row['ML in Job']
            # print(row_MLinJob)
            row_MLinJob = row_MLinJob.strip()
            if (row_MLinJob == "I do not know" or row_MLinJob == "No (we do not use ML methods)"):
                OHE_MLinJob.append(0)
            elif (row_MLinJob == "We are exploring ML methods (and may one day put a model into production)"):
                OHE_MLinJob.append(1)
            elif (
                    row_MLinJob == "We use ML methods for generating insights (but do not put working models into production)"):
                OHE_MLinJob.append(2)
            elif (
                    row_MLinJob == "We recently started using ML methods (i.e., models in production for less than 2 years)"):
                OHE_MLinJob.append(3)
            elif (
                    row_MLinJob == "We have well established ML methods (i.e., models in production for more than 2 years)"):
                OHE_MLinJob.append(4)

        OHE_MLinJob = np.array(OHE_MLinJob)
        OHE_MLinJob = OHE_MLinJob.reshape(-1, 1)

        print("Shape of the OHE_MLinJob: ", OHE_MLinJob.shape)

        return OHE_MLinJob


# Ordinal Encoding for Spending on ML
class SpendingonMLTransformer:
    def fit(self, df, y=None, **fit_params):

        return df

    def transform(self, df, y=None, **fit_params):

        proportions = df['Spending on ML'].value_counts(normalize=True)
        unique_spendingonjob_ranges = proportions.index
        df['Spending on ML'] = df['Spending on ML'].fillna(
            pd.Series(np.random.choice(unique_spendingonjob_ranges, p=proportions, size=len(df))))

        # df['Spending on ML'] = df['Spending on ML'].fillna('$0 ($USD)')

        # print(df['Spending on ML'].value_counts(dropna=False))

        OHE_SpendinML = []

        for index, row in df.iterrows():
            row_SpendinML = row['Spending on ML']
            # print(row_MLinJob)
            row_SpendinML = row_SpendinML.strip()
            if (row_SpendinML == "$0 ($USD)"):
                OHE_SpendinML.append(0)
            elif (row_SpendinML == "$100,000 or more ($USD)"):
                OHE_SpendinML.append(100000)
            else:
                # print(row_SpendinML)
                row_SpendinML = row_SpendinML.replace("$", "")
                row_SpendinML = row_SpendinML.replace(",", "")
                row_SpendinML = row_SpendinML.strip()
                lower, upper = row_SpendinML.split("-")
                mean_spend = (float(lower) + float(upper)) / 2
                # print(mean_spend)
                OHE_SpendinML.append(mean_spend)

        OHE_SpendinML = np.array(OHE_SpendinML)
        min_val = np.min(OHE_SpendinML)
        max_val = np.max(OHE_SpendinML)
        OHE_SpendinML = (OHE_SpendinML - min_val) / (max_val - min_val)

        print(min_val)
        print(max_val)

        OHE_SpendinML = OHE_SpendinML.reshape(-1, 1)

        print("Shape of the OHE_SpendinML: ", OHE_SpendinML.shape)

        return OHE_SpendinML


def getTransformedData(df):
    print("Entered getTransformedData Function")

    featureTransformer = FeatureUnion([
        ('Experience', Pipeline([('exp', ExpTransformer())])),
        ('Age', Pipeline([('age', AgeTransformer())])),
        ('Education', Pipeline([('edu', EduTransformer())])),
        ('Programming_Languages', Pipeline([('PL', PLTransformer())])),
        ('Country', Pipeline([('Country', CountryTransformer())])),
        ('Data_Visualisation', Pipeline([('DV', DataVisTransformer())])),
        ('ML_Years', Pipeline([('ML_Years', MLYearsTransformer())])),
        ('Gender', Pipeline([('Gender', GenderTransformer())])),
        ('ML_in_Job', Pipeline([('ML_in_Job', MLinJobTransformer())])),
        ('Spending_on_ML', Pipeline([('Spending_on_ML', SpendingonMLTransformer())])),
    ])

    print("FeatureUnion Performed")
    featureTransformer.fit(df)
    print("FeatureTransformer Fit Performed")
    transformed_data = featureTransformer.transform(df)

    return transformed_data


def getUserProfile_FV(user_profile):
    user_profile_FV = []
   
    #Transforming the experience in the user profile 
    user_profile_exp = user_profile[0]
    if (user_profile_exp == "I have never written code"):
        user_profile_exp_num = 0
    elif (user_profile_exp == "< 1 years"):
        user_profile_exp_num = 1
    elif (user_profile_exp == "20+ years"):
        user_profile_exp_num = 20
    else:
        user_profile_exp = user_profile_exp.replace("years", "")
        user_profile_exp = user_profile_exp.strip()
        lower, upper = user_profile_exp.split("-")
        mean_exp = (float(lower) + float(upper)) / 2
        user_profile_exp_num = mean_exp

    user_profile_FV.append(user_profile_exp_num)

    #Transforming the age in the user profile
    user_profile_age = user_profile[1]
    if (user_profile_age == "70+"):
        user_profile_age_num = 75
    else:
        lower, upper = user_profile_age.split("-")
        mean_age = (float(lower) + float(upper)) / 2
        user_profile_age_num = mean_age

    min_val = 21
    max_val = 75
    user_profile_age = (user_profile_age_num - min_val) / (max_val - min_val)
    user_profile_FV.append(user_profile_age_num)    

    # Transforming the education in the user profile
    user_profile_edu_df = pd.DataFrame(0, index=range(1), columns=['Education'])
    user_profile_edu_df.at[0, 'Education'] = user_profile[2].strip()
    edt = EduTransformer()
    user_profile_edt = edt.transform(user_profile_edu_df)
    user_profile_FV.append(user_profile_edt[0, 0])

    # Transforming the PL in the user profile
    user_profile_PL_list = user_profile[3]
    pl_series = pd.Series(0, index=pl_list)
    for pl in user_profile_PL_list:
        pl_series[pl] = 1
    for item in pl_series:
        user_profile_FV.append(item)

    # Transforming the Country in the user profile
    user_profile_cn = user_profile[4]
    country_list_all_data = fetch_recommender_data()['Country']
    unique_country_list = country_list_all_data.unique()
    country_series = pd.Series(0, index=unique_country_list)
    country_series[user_profile_cn] = 1
    for item in country_series:
        user_profile_FV.append(item)

    # Transforming the DV in the user profile
    user_profile_dv = user_profile[5]
    user_profile_FV.append(len(user_profile_dv))

    # Transforming the ML Years in the user profile
    user_profile_MLinYears = user_profile[6]
    if (user_profile_MLinYears == "I do not use machine learning methods"):
        user_profile_MLinYears_num = 0
    elif (user_profile_MLinYears == "Under 1 year"):
        user_profile_MLinYears_num = 0.5
    elif (user_profile_MLinYears == "20 or more years"):
        user_profile_MLinYears_num = 20
    else:
        user_profile_MLinYears = user_profile_MLinYears.replace("years", "")
        user_profile_MLinYears = user_profile_MLinYears.strip()
        lower, upper = user_profile_MLinYears.split("-")
        mean_exp = (float(lower) + float(upper)) / 2
        user_profile_MLinYears_num = mean_exp
    min_val = 0
    max_val = 20
    user_profile_MLinYears_num = (user_profile_MLinYears_num - min_val) / (max_val - min_val)
    user_profile_FV.append(user_profile_MLinYears_num)

    # Transforming Gender in the user profile
    user_profile_Gender = user_profile[7]
    gender_list = fetch_recommender_data()['Gender'].unique()
    gender_series = pd.Series(0, index=gender_list)
    print(len(gender_series))
    print(gender_series)
    gender_series[user_profile_Gender] = 1
    print(len(gender_series))
    print(gender_series)
    for item in gender_series:
        user_profile_FV.append(item)

    # Transforming ML in Job in the user profile
    user_profile_MLinJob = user_profile[8]
    user_profile_MLinJob = user_profile_MLinJob.strip()
    user_profile_MLinJob_num = 0
    if (user_profile_MLinJob == "I do not know" or user_profile_MLinJob == "No (we do not use ML methods)"):
        user_profile_MLinJob_num = 0
    elif (user_profile_MLinJob == "We are exploring ML methods (and may one day put a model into production)"):
        user_profile_MLinJob_num = 1
    elif (
            user_profile_MLinJob == "We use ML methods for generating insights (but do not put working models into production)"):
        user_profile_MLinJob_num = 2
    elif (
            user_profile_MLinJob == "We recently started using ML methods (i.e., models in production for less than 2 years)"):
        user_profile_MLinJob_num = 3
    elif (
            user_profile_MLinJob == "We have well established ML methods (i.e., models in production for more than 2 years)"):
        user_profile_MLinJob_num = 4
    user_profile_FV.append(user_profile_MLinJob_num)

    # Transforming Spending on ML in the user profile
    user_profile_SpendingonML = user_profile[9]
    user_profile_SpendingonML = user_profile_SpendingonML.strip()
    if (user_profile_SpendingonML == "$0 ($USD)"):
        user_profile_SpendingonML_num = 0
    elif (user_profile_SpendingonML == "$100,000 or more ($USD)"):
        user_profile_SpendingonML_num = 100000
    else:
        user_profile_SpendingonML = user_profile_SpendingonML.replace("$", "")
        user_profile_SpendingonML = user_profile_SpendingonML.replace(",", "")
        user_profile_SpendingonML = user_profile_SpendingonML.strip()
        lower, upper = user_profile_SpendingonML.split("-")
        mean_spend = (float(lower) + float(upper)) / 2
        user_profile_SpendingonML_num = mean_spend
    min_val = 0
    max_val = 100000
    user_profile_SpendingonML_num = (user_profile_SpendingonML_num - min_val) / (max_val - min_val)
    user_profile_FV.append(user_profile_SpendingonML_num)

    user_profile_FV = np.array(user_profile_FV)
    user_profile_FV = user_profile_FV.reshape(-1, 1)
    user_profile_FV = user_profile_FV.T

    return user_profile_FV

@st.cache_data
def get_random_forest_model():
    #Encoding the X_train and X_test so that they can be used to train and test the model.
    y_labels = fetch_recommender_data()['Job Role']

    X_train, X_test, y_train, y_test = train_test_split(fetch_recommender_data(), y_labels, test_size=0.2, random_state=42)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(X_train.iloc[3])

    X_train_encoded = getTransformedData(X_train)
    print(X_train_encoded.shape)

    X_test_encoded = getTransformedData(X_test)
    print(X_test_encoded.shape)

    print(X_train_encoded[3])

    #Random Forest Classification
    rf_classifier_model = RandomForestClassifier(random_state=42)
    rf_classifier_model.fit(X_train_encoded, y_train)

    rf_result = rf_classifier_model.predict(X_test_encoded)
    rf_accuracy = accuracy_score(y_test, rf_result)
    print(rf_accuracy)
    print(type(rf_classifier_model))

    return rf_classifier_model;
