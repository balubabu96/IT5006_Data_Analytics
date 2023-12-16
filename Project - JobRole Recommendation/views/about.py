import streamlit as st


class About:
    class Model:
        pageTitle = "Authors"

    def view(self, model):
        st.markdown("---")
        st.subheader(model.pageTitle)
        with st.container():
            col1, col2 = st.columns(2)

            with col1:

                st.markdown("- Zheng Weiliang - A0227418A")
                st.markdown("- Balu Babu - A0215463J")
                st.markdown("- Sivakumar Lakshminarayanan - A0275730")

                st.markdown('''
                <style>
                [data-testid="stMarkdownContainer"] ul{
                    padding-left:40px;
                }
                </style>
                ''', unsafe_allow_html=True)
