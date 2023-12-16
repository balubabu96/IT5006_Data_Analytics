import streamlit as st
import tools.utilities as util
import pandas as pd
import math
import statistics
import numpy as np
from PIL import Image

class Recommender:
    class Model:
        pageTitle = "Job Role Recommender"

    def view(self, model):
        st.markdown("---")
        st.subheader(model.pageTitle)

        data = util.fetch_recommender_data()

        # Sorted option to form fields
        country_list_all_data = data['Country']
        unique_country_list = country_list_all_data.unique()

        #STREAMLIT APP CONSTRUCTION

        tab1, tab2, tab3 = st.tabs(["Group", "User", "How"])

        with tab1:
            st.subheader("Group")
            uploaded_file = st.file_uploader("Choose a file to get recommendation for your profile")
            if uploaded_file is not None:
                if st.button('Run Recommendation'):
                    test_data_df = pd.read_csv(uploaded_file)
                    test_data_df_encoded = util.getTransformedData(test_data_df)

                    rf_result_test_data = util.get_random_forest_model().predict(test_data_df_encoded)
                    test_data_df['Recommended Job'] = rf_result_test_data

                    result_csv = test_data_df.to_csv(index=False)
                    st.markdown(util.generate_download_link(result_csv, "your_data.csv"), unsafe_allow_html=True)

        with tab2:
            st.subheader("User")
            st.write("Enter the following details of your profile to get a job role recommendation")

            age_range_values = ['18-24','25-29','50-54','55-59','35-39','60-69','30-34','40-44','45-49','70+']
            Education_values = ['No formal education past high school','Some college/university study without earning a bachelor’s degree','Bachelor’s degree','Master’s degree','Professional degree','Doctoral degree','Professional doctorate','I prefer not to answer']
            Experience_values = ['I have never written code','< 1 years','1-3 years','3-5 years','20+ years','5-10 years','10-20 years']
            ML_Years = ['I do not use machine learning methods','Under 1 year','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-20 years','20 or more years']
            ML_in_Job = ['No (we do not use ML methods)',
            'We are exploring ML methods (and may one day put a model into production)',
            'We use ML methods for generating insights (but do not put working models into production)',
            'We recently started using ML methods (i.e., models in production for less than 2 years)'
            'We have well established ML methods (i.e., models in production for more than 2 years)',
            'I do not know']
            Spend_on_ML = ['$0 ($USD)','$1-$99', '$100-$999' , '$1000-$9,999' ,'$10,000-$99,999', '$100,000 or more ($USD)']

            Gender = st.selectbox("Gender", options=['Woman', 'Man', 'Others'])
            Age_Range = st.selectbox("Age_Range", options= sorted(age_range_values))
            Education = st.selectbox("Your highest level of qualification", options= Education_values)
            Experience = st.selectbox("Years of coding experience", options= Experience_values)
            Country = st.selectbox("Country", options= sorted(unique_country_list))
            PL = st.multiselect("What programming languages are in proficient in?", options=sorted(util.pl_list))
            DV = st.multiselect("What data visualisation tools are you proficient in?", options=sorted(util.dv_tools_list))
            ML_Years = st.selectbox("How many years of experience of machine learning do you have?", options = ML_Years)
            ML_in_Job = st.selectbox("To what extent is your employer using machine learning?", options = ML_in_Job)
            Spend_on_ML = st.selectbox("How much did you or your team spend on ML in the last 2 years?", options= sorted(Spend_on_ML))        

            def getIncomeforRole(Job_Role):

                proportions = data['Income'].value_counts(normalize=True)
                unique_income_ranges = proportions.index
                data['Income'] = data['Income'].fillna(pd.Series(np.random.choice(unique_income_ranges,p=proportions,size=len(data))))
                
                print("Entered getIncomeForRole function")
                print("The job role for which income has to be determined is", Job_Role)
                job_role_income = data[data['Job Role'] == Job_Role]['Income']
                print(len(job_role_income))
                job_role_income = [x for x in job_role_income if x is not None]
                print(len(job_role_income))
                # print(job_role_income)
                income_num = []
                for item in job_role_income:
                    print(item)
                    if (item == "$0-999"):
                        income_num.append(500)
                    elif (item == "> $500,000"):
                        income_num.append(500000)
                    elif (type(item) == "str"):
                        item = item.replace(",", "")
                        item = item.strip()
                        lower, upper = item.split("-")
                        mean_income = (float(lower) + float(upper)) / 2
                        income_num.append(mean_income)

                median_income = statistics.median(income_num)
                #st.write("The median income for this role: ", median_income)
                st.write("Average income for this recommended role from available data is: ", sum(income_num) / len(income_num))

            def recommend(user_profile):

                result_FV = util.getUserProfile_FV(user_profile)
                pred_result = util.get_random_forest_model().predict(result_FV)



                #st.markdown(f"##The recommended job role for your profile is: **{ pred_result[0] }**")

                st.markdown(f"<h1 style='text-align: left; color: black; font-size: 30px;'>The recommended job role for your profile is: { pred_result[0] }</h1>",unsafe_allow_html=True)

                return pred_result[0];


            if st.button('Get Recommendation'):

                user_profile = []
                user_profile.append(Experience)
                user_profile.append(Age_Range)
                user_profile.append(Education)
                user_profile.append(PL)
                user_profile.append(Country)
                user_profile.append(DV)
                user_profile.append(ML_Years)
                user_profile.append(Gender)
                user_profile.append(ML_in_Job)
                user_profile.append(Spend_on_ML)

                recommended_job = recommend(user_profile)
                print("recommended_job", recommended_job)
                getIncomeforRole(recommended_job)
                # st.write(user_profile)
                print("Age Range", type(Age_Range))
                print("PL", type(PL))

        with tab3:
            st.markdown("**Steps Followed**")
            image = Image.open('images/banner.jpeg')
            st.image(image)
