import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from tools.utilities import fetch_processed_exploratory_data
from tools.utilities import fetch_processed_data_for_timetrend

class Stats:
    class Model:
        pageTitle = "Stats"
        timetrend_data = fetch_processed_data_for_timetrend()
        exploratory_data = fetch_processed_exploratory_data()

    def view(self, model):
        st.markdown("---")

        tab1, tab2 = st.tabs(["Trend over time", "Statistical tendencies"])

        with tab1:
            data = model.timetrend_data
            st.subheader("Trend over time")

            platform_list = ['Coursera','edX','Kaggle Learn Courses','DataCamp','Fast.ai','Udacity','Udemy','LinkedIn Learning','Cloud-Cert programs','University Courses','None','Other']
            needed_df = pd.DataFrame(index=[2020, 2021, 2022])

            selected_platform = st.multiselect('Select Platform ', platform_list)
            selected_countries = st.multiselect('Select Countries', data['Country'].unique())
            selected_age = st.multiselect('Select Age Range', data['Age'].unique())

            platform_list = selected_platform if selected_platform else platform_list
            filtered_df = data[data['Country'].isin(selected_countries)] if selected_countries else data
            filtered_df = filtered_df[filtered_df['Age'].isin(selected_age)] if selected_age else filtered_df

            for platform in platform_list:
                df = filtered_df[filtered_df[platform]==platform]
                df['Year'].astype(int)
                trend_platform = df.groupby('Year')[platform].count()
                needed_df[platform] = trend_platform

            st.table(needed_df)
            st.line_chart(needed_df.set_index(needed_df.index))
            st.write('<span style="font-size: 16px">X-Axis --> Years 2020 - 2022 </span>', unsafe_allow_html=True)
            st.write('<span style="font-size: 16px">Y-Axis --> Respondents Count</span>', unsafe_allow_html=True)        

        with tab2:
            data = model.exploratory_data
            st.subheader("Statistical tendencies")
            with st.container():

                col1, col2 = st.columns([2, 2])

                with col1:

                    col1.markdown("**By Job Role**")

                    sliced_data = data[['Year','Age','Experience','Education','Job Role']]

                    col1_selected_years = col1.multiselect('Select year(s)', sorted(sliced_data['Year'].unique()))
                    col1_selected_exp = col1.multiselect('Select Experience', sorted(sliced_data['Experience'].unique()))
                    col1_selected_job_role = col1.multiselect('Select Job Role', sorted(sliced_data['Job Role'].unique()))

                    col1_filtered_df = sliced_data[sliced_data['Year'].isin(col1_selected_years)] if col1_selected_years else sliced_data
                    col1_filtered_df = col1_filtered_df[col1_filtered_df['Experience'].isin(col1_selected_exp)] if col1_selected_exp else col1_filtered_df
                    col1_filtered_df = col1_filtered_df[col1_filtered_df['Job Role'].isin(col1_selected_job_role)] if col1_selected_job_role else col1_filtered_df

                    age_group = col1_filtered_df.groupby('Age')['Age'].count()
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.pie(age_group, labels=age_group.index,autopct='%1.1f%%')
                    ax.axis('equal')              
                    col1.table(age_group.to_frame().T.transpose())
                    col1.pyplot(fig)

                with col2:

                    col2.markdown("**By Age**")

                    sliced_data = data[['Year','Age','Experience','Education','Job Role','Yearly Salary']]

                    col2_selected_years = col2.multiselect('Select year(ss)', sorted(sliced_data['Year'].unique()))
                    col2_selected_Education = col2.multiselect('Select Education', sorted(sliced_data['Education'].unique()))
                    col2_selected_sal_range = col2.multiselect('Select Salary Range', sorted(sliced_data['Yearly Salary'].unique()))

                    col2_filtered_df = sliced_data[sliced_data['Year'].isin(col2_selected_years)] if col2_selected_years else sliced_data
                    col2_filtered_df = col2_filtered_df[col2_filtered_df['Education'].isin(col2_selected_Education)] if col2_selected_Education else col2_filtered_df
                    col2_filtered_df = col2_filtered_df[col2_filtered_df['Yearly Salary'].isin(col2_selected_sal_range)] if col2_selected_sal_range else col2_filtered_df

                    role_group = col2_filtered_df.groupby('Job Role')['Job Role'].count()
                    role_group = role_group.sort_values(ascending = False)

                    fig, ax = plt.subplots()
                    ax.bar(role_group.index, role_group.values, color='green')
                    ax.set_xlabel('Job Roles')
                    ax.set_ylabel('Count')
                    ax.set_xticklabels(role_group.index, rotation=90)
                    col2.table(role_group.to_frame().T.transpose())
                    col2.pyplot(fig)
