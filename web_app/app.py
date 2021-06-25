#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:39:53 2021

@author: saurabh
"""

import predict
import analysis
import streamlit as st
PAGES = {
    "Predict": predict,
    "Analysis": analysis
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.main()