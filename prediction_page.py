import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import plotly.express as px
import json
from streamlit_plotly_events import plotly_events
import itertools
import pickle
import joblib
import subprocess
from header import render_header




if True:
    number_words = {
        "a": 1, "one": 1, "single": 1,
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "twentyone": 21, "twenty one": 21,
        "twentytwo": 22, "twenty two": 22,
        "twentythree": 23, "twenty three": 23,
        "twentyfour": 24, "twenty four": 24,
        "twentyfive": 25, "twenty five": 25,
        "twentysix": 26, "twenty six": 26,
        "twentyseven": 27, "twenty seven": 27,
        "twentyeight": 28, "twenty eight": 28,
        "twentynine": 29, "twenty nine": 29,
        "thirty": 30, "thirtyone": 31, "thirty one": 31,
        "thirtytwo": 32, "thirty two": 32,
        "thirtythree": 33, "thirty three": 33,
        "thirtyfour": 34, "thirty four": 34,
        "thirtyfive": 35, "thirty five": 35,
        "thirtysix": 36, "thirty six": 36,
        "thirtyseven": 37, "thirty seven": 37,
        "thirtyeight": 38, "thirty eight": 38,
        "thirtynine": 39, "thirty nine": 39,
        "forty": 40, "fortyone": 41, "forty one": 41,
        "fortytwo": 42, "forty two": 42,
        "fortythree": 43, "forty three": 43,
        "fortyfour": 44, "forty four": 44,
        "fortyfive": 45, "forty five": 45,
        "fortysix": 46, "forty six": 46,
        "fortyseven": 47, "forty seven": 47,
        "fortyeight": 48, "forty eight": 48,
        "fortynine": 49, "forty nine": 49,
        "fifty": 50
    }

    def clean_term_prediction(term):
        # print(term[0])
        term = str(term[0]).strip().lower()
        # print(term)

        if 'yr' in term or 'year' in term or 'years' in term or 'y' in term or 'yrs.' in term:

            parts = term.split()
            if parts[0].isdigit():
                return str(int(parts[0])*12)
            elif type(number_words[parts[0]])==int:
                return str(int(number_words[parts[0]]) * 12)
            else:
                '12'

        elif 'mo' in term or 'month' in term or 'months' in term or 'mth' in term or 'mths' in term or 'mnth' in term:
            parts = term.split()
            if parts[0].isdigit():
                return str(int(parts[0]))
            elif type(number_words[parts[0]])==int:
                return str(int(number_words[parts[0]]))
            else:
                '12'
        elif 'week' in term or 'weeks' in term or 'wk' in term:
            parts = term.split()
            if parts[0].isdigit():
                return str(int(parts[0]/4))
            elif type(number_words[parts[0]])==int:
                return str(int(number_words[parts[0]]/4))
            else:
                '12'
        return '12'


    def clean_port_speed_prediction(value):
        if isinstance(value, list):
            value = value[0]  
        
        if pd.isna(value):
            return None

        value = str(value).strip().lower()  # Ensure it's a string

        numbers = [float(n) for n in re.findall(r"[\d\.]+", value)]

        if not numbers:
            return None

        max_value = max(numbers)

        if "g" in value:
            max_value *= 1000  # Convert Gbps to Mbps

        return int(max_value)
    

    # for displaying uniform term values
    def display_normal_term(val):
        val = val.lower().replace(" ", "")
        # Handle word numbers
        for word, num in number_words.items():
            if val.startswith(word):
                val = val.replace(word, str(num))
                break

        match = re.search(r'(\d+)\s*(\w*)', val)
        if match:
            num, unit = match.groups()
            if 'month' in unit or 'mth' in unit:
                return f"{num} months"
            elif 'yr' in unit or 'year' in unit:
                return f"{num} years" if int(num) > 1 else f"{num} year"
            else:
                return f"{num} months"
        else:
            return val


    #text box
    st.markdown("---")
    st.subheader("Add New Common Text")

    new_text = st.text_input("Enter text to add to common_text.txt:", placeholder="Type here...")

    if st.button("Submit"):
        if new_text.strip() == "":
            st.warning("Please enter some text before submitting.")
        else:

            with open("NER/common_text.txt", "w") as f:
                f.write(new_text.strip())

            st.success("Text saved successfully to NER/common.txt ✅")

            result = subprocess.run(
                ["python3", "NER/final_implementation.py"], 
                capture_output=True, text=True
            )

            if result.returncode == 0:
                st.success("Successfully ran final_implementation.py 🚀")
            else:
                st.error(f"Error running script: {result.stderr}")







    #check boxes for final implementation
    with open('NER/extracted_data.json', 'r') as json_data:
        d = json.load(json_data)

    st.title("MRC Prediction Graph")
    left, right = st.columns([1, 3])

    selected_values = {}

    with left:
        st.header("Select Entities")

        for key, value_list in d.items():
            st.subheader(f"{key}")
            selected = []
            
            # Clean the display values
            display_values = []
            for v in value_list:
                val = v[0] if isinstance(v, list) else v  # handle [value, tag] case

                if key == 'PORT_SPEED':
                    # Clean up port speed formats
                    match = re.search(r'(\d+\.?\d*)\s*(\w+)', val.lower())
                    if match:
                        num, unit = match.groups()
                        if 'g' in unit:
                            clean_val = f"{num} gbps"
                        else:
                            clean_val = f"{num} mbps"
                    else:
                        clean_val = val 
                    display_values.append(clean_val)
                else:
                    display_values.append(val)

            for idx, val in enumerate(display_values):
                is_checked = st.checkbox(f"{val}", key=f"{key}_{val}", value=(idx == 0))
                if is_checked:
                    selected.append(val)

            if not selected:
                st.warning(f"Please select at least one {key}!", icon="⚠️")
            
            selected_values[key] = selected

    # Create combinations only based on selected values
    filtered_lists = [selected_values[key] for key in d.keys()]
    filtered_combinations = list(itertools.product(*filtered_lists))
    df = pd.DataFrame(filtered_combinations, columns=d.keys())
    temp_df = df.copy()



    # right part
    with right:
        # preprocessing
        df['TERM'] = df['TERM'].apply(clean_term_prediction)
        df['PORT_SPEED'] = df['PORT_SPEED'].apply(clean_port_speed_prediction)

        column_mapping = {
        'STATES_CITIES': 'A Loc State',
        'CIR_TYPES': 'generalized_Cir'
        }
        df.rename(columns=column_mapping, inplace=True)



        with open("model_and_encoders/label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        for col in ['A Loc State', 'generalized_Cir']:
            if col in df.columns:
                df[col + '_Freq'] = df[col].map(
                    lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
                )

        removed_indexes = df[df['A Loc State_Freq'] == -1].index.tolist()

        new_test_data = df[df['A Loc State_Freq'] != -1]

        df.drop(columns=['A Loc State', 'generalized_Cir'], inplace=True)





        provider_count = pd.read_csv('provider_counts.csv')
        provider_encoding = pd.read_csv('provider_encoding.csv')
        provider_encoding.drop(columns=['Unnamed: 0'], inplace=True)

        provider_list = list(provider_encoding.iloc[:,0])
        provider_encoding_list = list(provider_encoding.iloc[:,1])
        provider_count_list = list(provider_count.iloc[:,0])

        #slider
        num_providers = st.slider('Select number of providers to display', min_value=1, max_value=15, value=5)
        
        extract_provider_indexes = []

        for idx in range(num_providers):
            temp_provider = provider_count_list[idx]
            index = provider_list.index(temp_provider)
            extract_provider_indexes.append(provider_encoding_list[index])





        for i in range(num_providers):
            df[provider_count_list[i]] = extract_provider_indexes[i]

        df['TERM'] = pd.to_numeric(df['TERM'], errors='coerce').astype('Int64')

        temp_data = pd.DataFrame()
        best_xgb= joblib.load("./model_and_encoders/best_xgb_model.pkl")

        # prediction
        for i in range(num_providers):
            features_inorder = [
            'PORT_SPEED', 'TERM', 'A Loc State_Freq', 
            provider_count_list[i],
            'generalized_Cir_Freq']

            X_test = df[features_inorder]

            y_pred = best_xgb.predict(X_test)

            temp_data[provider_count_list[i]] = y_pred
        


        # highlighting the lowest
        def highlight_min(s):
            is_min = s == s.min()
            return ['background-color: red' if v else '' for v in is_min]

        styled_df = temp_data.style.apply(highlight_min, axis=1)


        # Merge dataframes
        merged_df = pd.concat([temp_df.reset_index(drop=True), styled_df.data.reset_index(drop=True)], axis=1)
        st.dataframe(merged_df, use_container_width=True)



        # Plotting graphs
        provider_list = temp_data.columns.tolist()
        temp_data_long = temp_data.copy()
        temp_data_long['Combination'] = [f'Combo {i+1}' for i in range(len(temp_data))]
        temp_data_long = temp_data_long.melt(id_vars='Combination', var_name='Provider', value_name='Price')

        # Create the line plot
        fig = px.line(temp_data_long, 
                    x='Provider', 
                    y='Price', 
                    color='Combination',
                    markers=True,
                    title="Predicted Prices for Different Combinations",
                    labels={'Price': 'Price in dollars'},
                    )

        fig.update_layout(
            xaxis_tickangle=45,
            legend_title="Combinations",
            legend=dict(x=1.05, y=1),
            width=1000,
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)



else:
    st.info("👆 Upload a CSV file to begin analysis.")
