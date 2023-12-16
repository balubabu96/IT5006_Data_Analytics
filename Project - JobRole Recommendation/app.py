import streamlit as st

from streamlit_option_menu import option_menu
from tools.utilities import load_css
from views.project import Project
from views.recommender import Recommender
from views.stats import Stats
from views.about import About

load_css()

class Model:
    menuTitle = "IT5006 Project"
    option1 = "Requirements"
    option2 = "Exploratory"
    option3 = "Recommendation"
    option4 = "About"

    menuIcon = "menu-up"
    icon1 = "speedometer"
    icon2 = "clipboard-data"
    icon3 = "clipboard-data"
    icon4 = "chat"

def view(model):
    with st.container():
        menuItem = option_menu(None,
                               [model.option1, model.option2, model.option3, model.option4],
                               icons=[model.icon1, model.icon2, model.icon3, model.icon3],
                               menu_icon=model.menuIcon,
                               default_index=0,
                               orientation="horizontal", 
                               styles={
                                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                                    "icon": {"color": "orange", "font-size": "25px"}, 
                                    "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                    "nav-link-selected": {"background-color": "green"},
                                })

    if menuItem == model.option1:
        Project().view(Project.Model())

    if menuItem == model.option2:
        Stats().view(Stats.Model())

    if menuItem == model.option3:
        Recommender().view(Recommender.Model())
    
    if menuItem == model.option4:
        About().view(About.Model())

view(Model())
