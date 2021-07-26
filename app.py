'''        ------------------ MARCO POLO ------------------
               --------< Author - Aditya Shenoy >--------
'''

# Core Packages
import streamlit as st
import streamlit.components.v1 as stc

# Util packages
import sys 

# Additional Packages
import pandas as pd
import numpy as np
from pandasql import sqldf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# Avoiding Traceback calls


# from streamlit.state.session_state import SessionState
# sys.tracebacklimit=0




# Page Layout Configuration
PAGE_CONFIG = {'page_title':'Marco Polo','layout':'wide'}
st.set_page_config(**PAGE_CONFIG)

# App Functions

# Formatting and Styling Functions
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def title_html(title):
    title_html = '''
            <p style="
                border: #f63366;
                border-style: double;
                border-radius: 25px;
                text-align: center;
                font-size: 130%;
                border-width: thick;
                ">
            {}
            </p>
            '''
    return title_html.format(title)


def value_html(text):
    value_html = '''
            <p style="
                border: #f63366;
                border-style: inset;
                font-family: monospace;
                text-align: center;
                border-width: thin;
                border-radius: 10px;
                ">
            {}
            </p>
            '''
    return value_html.format(text)



pysqldf = lambda q : sqldf(q,globals())

# Main Function
def main():
    # Loading the CSS styling file
    local_css("style.css")
    
    # App Title
    st.title('Marco Polo')
    # File Uploader
    dataset = st.sidebar.file_uploader('Upload Dataset',type=['csv'])
    
    if dataset is not None:
        data = pd.read_csv(dataset)
        numeric_columns = list(data.select_dtypes(include=['float64','int']).columns)
        string_columns = list(data.select_dtypes(include=['object','category']).columns)
        date_columns = list(data.select_dtypes(include=['datetime64','timedelta']).columns)
    # Sidebar - Menu
    Menu = ['Home','SQL Playground']
    choice = st.sidebar.selectbox('Menu',Menu)
    

    if choice == 'Home':
        if dataset is not None:
            st.markdown(title_html(dataset.name),unsafe_allow_html=True)
            st.markdown('<br>',unsafe_allow_html=True)
            c1,c2 = st.beta_columns([9,1])
            with c1:
                num = st.slider('Select no of rows to display',5,50,10,step=5)
            with c2:
                st.markdown('<br>',unsafe_allow_html=True)
                st.write("")
                all_button = st.button('Show all')
            if all_button:
                st.dataframe(data)
            else:
                st.dataframe(data.head(num))

            #File Details
            File_details = st.beta_expander('File Details')
            with File_details:
                file_details = {
                "File name": dataset.name,
                "File type" : dataset.type,
                "File size" : str(dataset.size) + ' bytes'}
                for key,value in file_details.items():
                    st.write("{} : {}".format(key,value))

            # Initial Data Summary
            col1, col2, col3 = st.beta_columns([1,1,2])

            # Dataset Shape
            with col1:
                expand1 = st.beta_expander('Dataset Shape')
                with expand1:
                    rows,columns = data.shape
                    st.write('Number of rows : {}'.format(rows))
                    st.write('Number of rows : {}'.format(columns))

            # Dataset Datatypes
            with col2:
                expand2 = st.beta_expander('Dataset Datatypes')
                with expand2:
                    dtype_df = data.dtypes
                    # To avoid string break with long index names
                    dtype_df.index = np.where(dtype_df.index.str.len()<15,dtype_df.index,dtype_df.index.str[:12]+'...')
                    dtype_df.name = 'Datatypes'
                    st.dataframe(dtype_df)
                    
            # Dataset Info
            with col3:
                expand3 = st.beta_expander('Dataset Info')
                with expand3:
                    st.dataframe(data.describe())

            
            # Column-wise Descriptive Summary

            Desc_summary_expander = st.beta_expander('Univariate Analysis')
            with Desc_summary_expander:
                
                # Selecting a column
                attribute = st.selectbox('Select a Column',numeric_columns+string_columns+date_columns,key='uni_var_button')
                
                b1 = st.button('Summarize')
                st.markdown('<br>',unsafe_allow_html=True)

                if b1:
                    numeric_cont = st.beta_container()
                    string_cont = st.beta_container()
                    date_cont = st.beta_container()

                    #Checking the selected column datatype
                    if attribute in numeric_columns:
                        with numeric_cont:
                            c1,c2,c3 = st.beta_columns([1,2,2])
                            with c1:
                                st.write(data[attribute].dtypes)
                                st.write(data[attribute].describe())

                                # Calculating Distinct Values
                                distinct_values = data[attribute].nunique()
                                distinct_values_str = 'Distinct Values: ' + str(distinct_values)
                                st.markdown(value_html(distinct_values_str),unsafe_allow_html=True)

                                # Calculating Null Values
                                null_values = data[attribute].isnull().sum()
                                null_values_str = 'Missing Values: ' + str(null_values)
                                st.markdown(value_html(null_values_str),unsafe_allow_html=True)

                                # Calculating Missing Value %
                                missing_value_perc = (data[attribute].isnull().sum()/data[attribute].count())*100
                                missing_value_perc_str = 'Missing Value %: ' + str(missing_value_perc) + '%'
                                st.markdown(value_html(missing_value_perc_str),unsafe_allow_html=True)

                                # Outlier Detection
                                threshold = 3
                                outlier = []
                                attri_mean = data[attribute].mean()
                                attri_std = data[attribute].std()
                                for val in data[attribute].values:
                                    z = (val-attri_mean)/attri_std
                                    if z >  threshold:
                                        outlier.append(val)
                                outlier_perc = (len(outlier)/data[attribute].count())*100

                                outlier_str = 'No of outliers: ' + str(len(outlier))
                                outlier_perc_str = 'Outlier %: ' + str(outlier_perc) + '%'
                                st.markdown(value_html(outlier_str),unsafe_allow_html=True)
                                st.markdown(value_html(outlier_perc_str),unsafe_allow_html=True)

                                # Skewness
                                skewness = data[attribute].skew().round(4)
                                skewness_str = 'Skewness: ' + str(skewness)
                                st.markdown(value_html(skewness_str),unsafe_allow_html=True)

                            with c2:
                                # Boxplot
                                st.markdown(title_html('Boxplot'),unsafe_allow_html=True)
                                fig11, ax11 = plt.subplots()
                                ax11 = sns.boxplot(y=data[attribute],color='#f63366')
                                ax11.set_ylabel(attribute, fontsize=10, fontweight='bold')
                                st.pyplot(fig11)

                                # Line
                                st.markdown(title_html('HistPlot'),unsafe_allow_html=True)
                                fig12, ax12 = plt.subplots() 
                                #ax12 = plt.plot(data[attribute],marker='o',color='#f63366',markerfacecolor='#04022d')
                                ax12 = sns.histplot(data=data, x=data[attribute], color='#f63366')
                                ax12.set_xlabel(attribute, fontsize=10, fontweight='bold')
                                ax12.set_ylabel('Count', fontsize=10, fontweight='bold')
                                st.pyplot(fig12)
                                

                               


                            with c3:
                                # Stripplot
                                st.markdown(title_html('Stripplot'),unsafe_allow_html=True)
                                fig13, ax13= plt.subplots()
                                ax13 = sns.stripplot(y=data[attribute], color='#04022d',edgecolor='#f63366',linewidth=1.5)
                                ax13.set_ylabel(attribute, fontsize=10, fontweight='bold')
                                st.pyplot(fig13)
                                
                                # KDE Distplot
                                st.markdown(title_html('KDE Plot'),unsafe_allow_html=True)
                                fig14, ax14= plt.subplots()
                                ax14 = sns.kdeplot(data=data, x= data[attribute], color = '#f63366',fill=True)
                                ax14.set_xlabel(attribute, fontsize=10, fontweight='bold')
                                ax14.set_ylabel('Density', fontsize=10, fontweight='bold')
                                st.pyplot(fig14)

                    if attribute in string_columns:
                        with string_cont:
                            c1,c2,c3 = st.beta_columns([1,2,2])
                            with c1:
                                st.write(data[attribute].dtypes)

                                # Calculating Distinct Values
                                distinct_values = data[attribute].nunique()
                                distinct_values_str = 'Distinct Values: ' + str(distinct_values)
                                st.markdown(value_html(distinct_values_str),unsafe_allow_html=True)

                                # Calculating Null Values
                                null_values = data[attribute].isnull().sum()
                                null_values_str = 'Missing Values: ' + str(null_values)
                                st.markdown(value_html(null_values_str),unsafe_allow_html=True)

                                # Calculating Missing Value %
                                missing_value_perc = round((data[attribute].isnull().sum()/(data[attribute].count()+data[attribute].isnull().sum()))*100,2)
                                missing_value_perc_str = 'Missing Value %: ' + str(missing_value_perc) + '%'
                                st.markdown(value_html(missing_value_perc_str),unsafe_allow_html=True)

                                
                                   
                            with c2:
                                # Category Countplot   
                                st.markdown(title_html('Countplot'),unsafe_allow_html=True)
                                fig21,ax21 = plt.subplots()
                                ax21 = sns.countplot(x=data[attribute],order = pd.value_counts(data[attribute]).iloc[:].index,palette='rocket')
                                ax21.set_xlabel(attribute, fontsize=10, fontweight='bold')
                                ax21.set_ylabel('Count', fontsize=10, fontweight='bold')
                                st.pyplot(fig21)
                            
                            with c3:
                                # % Bar plot
                                st.markdown(title_html('% Barplot'),unsafe_allow_html=True)
                                data_per = data[attribute].value_counts(normalize=True)                                
                                data_per_df = data_per.reset_index()
                                fig22,ax22 = plt.subplots()
                                ax22 = sns.barplot(x=data_per_df['index'],y=data_per_df[attribute], palette='rocket',order = pd.value_counts(data[attribute]).iloc[:].index)
                                ax22.set_xlabel(attribute, fontsize=10, fontweight='bold')
                                ax22.set_ylabel('Percentage', fontsize=10, fontweight='bold')
                                st.pyplot(fig22)
                                 


                    if attribute in date_columns:
                        with string_cont:
                            st.write(data[attribute].dtypes)     
                        

            Bivariate_expander = st.beta_expander('Bivariate Analysis')
            with Bivariate_expander:
                
                c1, c2 = st.beta_columns(2)
                with c1:
                    x = st.selectbox("Select 'X' column",string_columns+numeric_columns)
                    st.write("")
                    hue_filt = st.selectbox('Select filter/hue',[None]+string_columns)
                with c2:
                    y = st.selectbox("Select 'Y' column",string_columns+numeric_columns)

                b2 = st.button('Summarize',key='bi_var_button')
                st.markdown('<br>',unsafe_allow_html=True)

                if b2:
                    num_num_cont = st.beta_container()
                    num_obj_cont = st.beta_container()                 
                
                    if x in string_columns and y in string_columns:
                        st.info("Can't Plot with both string columns")

                    elif x in numeric_columns and y in numeric_columns:
                        with num_num_cont:
                            c1,c2,c3 = st.beta_columns([1,2,2])
                            if hue_filt is None:
                                filt = None 
                            else:
                                filt = data[hue_filt]

                            with c2:
                                # Scatter Plot
                                st.markdown(title_html('Scatter PLot'),unsafe_allow_html=True)
                                fig11,ax11 = plt.subplots()
                                ax11 = sns.scatterplot(data[x],data[y],hue=filt,color='#f63366',palette='rocket')
                                st.pyplot(fig11)
                            
                            with c3:
                                # KDE Plot
                                st.markdown(title_html('KDE PLot'),unsafe_allow_html=True)
                                fig12,ax12 = plt.subplots()
                                ax12 = sns.kdeplot(data[x],data[y],data=data,hue=filt, color = '#f63366',palette='rocket')
                                st.pyplot(fig12)

                    else:
                        with num_obj_cont:
                            c1,c2 = st.beta_columns(2)
                            filt_flag = 0
                            column_flag = 0

                            if hue_filt is None:
                                filt = None 
                            else:
                                filt = data[hue_filt]
                                if filt.nunique() <= 3 :
                                    st.error('Choose a filt with 3 or less categorical values')
                                    filt_flag += 1

                            if x in string_columns:
                                if data[x].nunique() > 10:
                                    st.error("Chosen 'X' column attribute has high no of categorical values. Please choose another appropriate field to analyze")
                                    column_flag += 1
                            if y in string_columns:
                                if data[x].nunique() > 10:
                                    st.error("Chosen 'Y' column attribute has high no of categorical values. Please choose another appropriate field to analyze")
                                    column_flag += 1
                            

                            if filt_flag == 0 and column_flag == 0:
                                with c1:
                                    # Bar Plot
                                    st.markdown(title_html('BarPLot'),unsafe_allow_html=True)
                                    fig11,ax11 = plt.subplots()
                                    ax11 = sns.barplot(data[x],data[y],palette='rocket',hue=filt)
                                    ax11.set_xlabel(x, fontsize=10, fontweight='bold')
                                    ax11.set_ylabel(y, fontsize=10, fontweight='bold')
                                    st.pyplot(fig11)

                                    # Strip Plot
                                    st.markdown(title_html('StripPLot'),unsafe_allow_html=True)
                                    fig12,ax12 = plt.subplots()
                                    ax12 = sns.stripplot(data[x],data[y],palette='rocket',hue=filt,jitter=True, dodge=True)
                                    ax12.set_xlabel(x, fontsize=10, fontweight='bold')
                                    ax12.set_ylabel(y, fontsize=10, fontweight='bold')
                                    st.pyplot(fig12)




                                with c2:
                                    # Box Plot
                                    st.markdown(title_html('BoxPLot'),unsafe_allow_html=True)
                                    fig21,ax21 = plt.subplots()
                                    ax21 = sns.boxplot(data[x],data[y],palette='rocket',hue=filt)
                                    ax21.set_xlabel(x, fontsize=10, fontweight='bold')
                                    ax21.set_ylabel(y, fontsize=10, fontweight='bold')
                                    st.pyplot(fig21)


                                    # Violin Plot
                                    st.markdown(title_html('ViolinPLot'),unsafe_allow_html=True)
                                    fig22, ax22 = plt.subplots()
                                    ax22 = sns.violinplot(data[x],data[y],palette='rocket',hue=filt)
                                    ax22.set_xlabel(x, fontsize=10, fontweight='bold')
                                    ax22.set_ylabel(y, fontsize=10, fontweight='bold')
                                    st.pyplot(fig22)
                            
                            
                                


                
                else:
                    pass


            

    elif choice == 'SQL Playground':
        if dataset is None:
            st.info('Upload a dataset to explore!')
        else:
            st.markdown(title_html('SQL Playground'),unsafe_allow_html=True)
            st.info("The data from the uploaded file is stored in 'data' table")
            st.markdown('<br>',unsafe_allow_html=True)
            c1,c2 = st.beta_columns([7,2])
            with c1:
                default_query = 'SELECT * FROM data'
                Query = st.text_area('Enter Query',default_query)
                try:    
                    output = sqldf(Query)
                    if output is not None:
                        st.write(output)
                except Exception as e:
                    st.write(e)
            with c2:
                table_header_html = '''
                    <p style="
                        border: #f63366;
                        border-style: dashed;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 130%;
                        border-width: medium;
                        ">
                    data
                    </p>
                    '''
                st.markdown(table_header_html,unsafe_allow_html=True)
                column_name_html = '''
                    <p style="
                        text-align:center;
                        font-family: monospace;
                        ">
                    {}
                    </p>
                '''
                for col in data.columns:
                    st.markdown(column_name_html.format(col),unsafe_allow_html=True)

       



        
if __name__ == '__main__':
    main()
