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





st.set_page_config(
    page_title="MRC Price Analysis Dashboard",
    layout="wide",
)

render_header()

st.markdown(
    f"""
    <style>
        .custom-offset {{
            margin-top: 0px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

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
        term = str(term[0]).strip().lower()

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

        value = str(value).strip().lower()

        numbers = [float(n) for n in re.findall(r"[\d\.]+", value)]

        if not numbers:
            return None

        max_value = max(numbers)

        if "g" in value:
            max_value *= 1000

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

    new_text = st.text_input("Enter the Place, Speed, Product, and Region below", placeholder="Type here...")

    if st.button("Submit"):
        if new_text.strip() == "":
            st.warning("Please enter some text before submitting.")
        else:

            with open("NER/common_text.txt", "w") as f:
                f.write(new_text.strip())

            st.success("Processing...")

            result = subprocess.run(
                ["python3", "NER/final_implementation.py"], 
                capture_output=True, text=True
            )

            if result.returncode == 0:
                st.success("Almost There...")
            else:
                st.error(f"Error running script: {result.stderr}")



    

    #check boxes for final implementation
    with open('NER/extracted_data.json', 'r') as json_data:
        d = json.load(json_data)

    st.title("Monthly Recurring Cost Estimation")
    left, right = st.columns([1, 3])

    selected_values = {}


    with left:
        st.header("Select Entities")

        for key, value_list in d.items():
            st.subheader(f"{key}")
            selected = []
            
            if key=="CIR_TYPES":
                if not value_list:
                    value_list = ['wavelength', 'evpl', 'epl', 'dia', 'broadband', 'darkfibre', 'e_line', 'mpls', 'microwave']
            # Clean the display values
            display_values = []
            for v in value_list:
                val = v[0] if isinstance(v, list) else v  

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
        num_providers = st.slider('Select number of providers', min_value=1, max_value=15, value=5)
        
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
        temp_data_long['Combination'] = [f'Combo {i}' for i in range(len(temp_data))]
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



    st.markdown("---")

    df = pd.read_csv('/Users/dhruv/Dhruv/Apcela/RFQ.csv')

    filter_state_data = df[df['MRC'] != 'No Bid'].copy()

    filter_state_data['MRC'] = filter_state_data['MRC'].str.replace(r'[\$,]', '', regex=True)
    filter_state_data['MRC'] = pd.to_numeric(filter_state_data['MRC'], errors='coerce')

    filter_state_data = filter_state_data[filter_state_data['MRC'] <= 20000]

    filter_state_data['A Loc City'] = filter_state_data['A Loc City'].str.lower()
    filter_state_data['A Loc State'] = filter_state_data['A Loc State'].str.lower()

    city_corrections = {
        '2-4-1, marunouchi,chiyoda-ku tokyo': 'tokyo',
        'cartaret': 'carteret',
        'chicago ': 'chicago',
        'frankfurt am main': 'frankfurt',
        'hongkong': 'hong kong',
        'kwai chung, nt': 'kwai chung',
        'munchen': 'munich',
        'new york city': 'new york',
        'nan jing xi lu, jingan qu, shanghai shi': 'shanghai',
        'pudong new area, shanghai': 'shanghai',
        'yangpu district, shanghai': 'shanghai',
        'rio de janerio': 'rio de janeiro',
        'rio de janerio/rj': 'rio de janeiro',
        'rmz ecoworld, campus 6b, 5th & part 6th floor (units 501 & 502 & 601)\nsarjapur, marathalli outer ring road \ndevarabeesanahalli village, varthur hobli\nbangalore east taluk \nbangalore – 560 103': 'bangalore',
        'sao paolo': 'sao paulo',
        'st. louis': 'st louis',
        'washington ': 'washington'
    }
    filter_state_data['A Loc City'] = filter_state_data['A Loc City'].replace(city_corrections)

    state_country_mapping = {
    'nj': 'nj', 'dc': 'dc', 'va': 'va', 'ks': 'ks', 'tx': 'tx', 'ga': 'ga', 'il': 'il', 'mn': 'mn', 
    'ny': 'ny', 'co': 'co', 'ct': 'ct', 'oh': 'oh', 'ca': 'ca', 'wa': 'wa', 'sc': 'sc', 'ut': 'ut',
    'pa': 'pa', 'fl': 'fl', 'ma': 'ma', 'ok': 'ok', 'mi': 'mi', 'tn': 'tn', 'nc': 'nc', 'wi': 'wi', 
    'az': 'az', 'or': 'or', 'mo': 'mo',

    'germany': 'germany', 'ireland': 'ireland', 'hong kong': 'hong kong', 'peru': 'peru',
    'mexico': 'mexico', 'india': 'india', 'china': 'china', 'russia': 'russia', 'south africa': 'south africa',
    'netherlands': 'netherlands', 'japan': 'japan', 'hk': 'hong kong', 'france': 'france', 
    'swizterland': 'switzerland', 'switzerland': 'switzerland', 'italy': 'italy', 'prc': 'china', 
    'canada': 'canada', 'london england': 'uk', 'uae': 'uae', 'ciudad de mexico': 'mexico', 
    'brasil': 'brazil', 'sweden': 'sweden', 'england': 'uk', 'australia': 'australia', 
    'alberta': 'canada', 'nicaragua': 'nicaragua', 'honduras': 'honduras', 'quebec': 'canada',
    'brazil': 'brazil', 'london': 'uk', 'jamaica': 'jamaica', 'switz.': 'switzerland', 'kow': 'hong kong',
    'in': 'india', 'poland': 'poland', 'czechia': 'czech republic', 'id': 'indonesia', 
    'new zealand': 'new zealand', 'united kingdom': 'uk', 'israel': 'israel', 'cyprus': 'cyprus', 
    'greece': 'greece', 'on, canada': 'canada', 'ab,canada': 'canada', 'united arab emirates': 'uae', 
    'romania': 'romania', 'qc': 'canada', 'ab': 'canada', 'spain': 'spain', 'colombia': 'colombia', 
    'nb': 'canada', 'quebec-canada': 'canada', 'lincolnshire': 'uk', 'santa catarina (sc)': 'brazil',
    'ontario - canada': 'canada', 'sp, brazil': 'brazil', 'beligium': 'belgium', 'pudong': 'china',
    'taipei': 'taiwan', 'bangalore':'india'
    }

    filter_state_data['A Loc State'] = filter_state_data['A Loc State'].str.strip().str.lower()
    filter_state_data['A Loc State'] = filter_state_data['A Loc State'].replace(state_country_mapping)

    def clean_term(term):
        term = str(term).strip().lower()

        if 'mtm' in term:
            return '12'
        elif 'co-term' in term or 'coterminous' in term:
            return '12'
        elif 'yr' in term or 'year' in term:
            parts = term.split()
            return str(int(parts[0]) * 12) if parts[0].isdigit() else '12'
        elif 'mo' in term or 'month' in term:
            return ''.join(filter(str.isdigit, term))
        elif term.isdigit():
            return term
        elif ',' in term:
            return term.split(',')[0].strip()

        return '12'



    filter_state_data['Term_Cleaned'] = filter_state_data['Term'].apply(clean_term)
    filter_state_data['Term_Cleaned'] = pd.to_numeric(filter_state_data['Term_Cleaned'], errors='coerce')




    def clean_port_speed(value):
        if pd.isna(value):
            return None

        value = value.strip().lower()


        numbers = [float(n) for n in re.findall(r"[\d\.]+", value)]

        if not numbers:
            return None

        max_value = max(numbers)


        if "g" in value:
            max_value *= 1000

        return int(max_value)


    data = pd.DataFrame({'Port Speed': [
        '100G', '1G', '5Mb', '10Mb', '20mB', '200Mb', '10G', '2.5G', '500',
        '100', '1000', '600', '100Mb', '100 Mbps', '20', '50', '100M',
        '100/25', '500/500', '1000/100', '1G Protected', '10G Wave', '10M',
        '100 Mbps', '10 Mbps', '1G CIR', '1G not 50Mb', 'P2P evpl', '10Gb'
    ]})

    filter_state_data['Port_Speed_Mbps'] = filter_state_data['Port Speed'].apply(clean_port_speed)

    overall_median_speed = filter_state_data['Port_Speed_Mbps'].median()
    filter_state_data['Port_Speed_Mbps'].fillna(overall_median_speed, inplace=True)

    class_conditions = {
        "Broadband": ["broadband"],  
        "DarkFiber": ["DF", "dark fiber"],  
        "DIA": ["DIA"],  
        "EVPL": ["EVPL", "Epath"],  
        "EPL": ["EPL", "EPLS", "Ethernet"],  
        "Wavelength": ["Wave"],  
        "Microwave": ["Microwave"],  
        "MPLS": ["MPLS"]  
    }

    def classify_text(text):
        if not isinstance(text, str):
            return "None"
        
        text = text.strip().lower()

        if re.match(r'^(epl|epls|ethernet)\b', text):
            return "EPL"
        
        if re.match(r'^wave\b', text):
            return "Wavelength"
        
        if re.match(r'^eline\b', text):
            return "E_Line"

        if "new wave/epl" in text:
            return "Wavelength"

        if "protected epl/wave" in text:
            return "EPL"

        if "mpls ethernet" in text:
            return "MPLS"

        if text == "1g broadband (quoted dia)":
            return "DIA"

        classes = set()
        for class_name, keywords in class_conditions.items():
            if any(re.search(rf'\b{kw.lower()}\b', text) for kw in keywords):
                classes.add(class_name)

        return ', '.join(sorted(classes)) if classes else "None"

    filter_state_data['generalized_Cir'] = filter_state_data['Cir Type'].apply(classify_text)

    filter_state_data = filter_state_data[filter_state_data['generalized_Cir']!='None']

    filter_state_data['generalized_Cir'] = filter_state_data['generalized_Cir'].astype(str).str.lower()

    filter_state_data.drop(columns=['Rec Num', 'RFQ', 'A Loc Address', 'A Loc City', 'A Loc Zip', 'Z Loc Address', 'Z Loc City', 'Z Loc State', 'Z Loc Zip', 'A Loc Local Loop Provider', 'NRC', 'RTD Latency', 'Cir Type', 'Port Speed', 'Term'], inplace=True)


    with open('state_mapping.json', 'r') as json_file:
        state_country_mapping = json.load(json_file)

    filter_state_data['A Loc State'] = filter_state_data['A Loc State'].map(state_country_mapping)
    filter_state_data['A Loc State'] = filter_state_data['A Loc State'].fillna(filter_state_data['A Loc State'])

    #preprocessing completed
    






    with st.container():
        col1, spacer, col2 = st.columns([6, 1, 7])




        with col1:
            # Plot 1
            st.header("Top Providers")

            top_x = st.slider("Number of Top Providers", min_value=1, max_value=20, value=5)

            top_providers = filter_state_data['Provider'].value_counts().nlargest(top_x).index
            df_filtered = filter_state_data[filter_state_data['Provider'].isin(top_providers)]


            fig1 = px.box(
                df_filtered,
                x='Provider',
                y='MRC',
                title=' ',
                color='Provider',
                points='all',
            )

            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=20, color='black'),
                xaxis_title='Provider',
                yaxis_title='MRC ($)',
                xaxis=dict(tickangle=45),
                font=dict(color='black'),
                showlegend=False,  # Hide legend since x-axis already labels providers
            )

            st.plotly_chart(fig1, use_container_width=True)





        with col2:
            # Plot 2
            st.header(" MRC vs Term")

            unique_states = filter_state_data['A Loc State'].dropna().unique()
            unique_states = sorted(unique_states)

            selected_states = st.multiselect(
                'Select States/Countries to display:',
                options=unique_states,
                default=['New Jersey', 'California', 'Washington DC']
            )

            filtered_data = filter_state_data[filter_state_data['A Loc State'].isin(selected_states)]

            city_term_mrc = filtered_data.groupby(['A Loc State', 'Term_Cleaned'])['MRC'].mean().reset_index()

            city_term_mrc['Term_Cleaned'] = city_term_mrc['Term_Cleaned'].astype(float)

            all_terms = sorted(city_term_mrc['Term_Cleaned'].unique())
            all_combinations = pd.MultiIndex.from_product(
                [selected_states, all_terms],
                names=['A Loc State', 'Term_Cleaned']
            ).to_frame(index=False)

            city_term_mrc_full = pd.merge(all_combinations, city_term_mrc, how='left', on=['A Loc State', 'Term_Cleaned'])
            city_term_mrc_full = city_term_mrc_full.sort_values(by=['A Loc State', 'Term_Cleaned'])

            fig = px.line(
                city_term_mrc_full,
                x='Term_Cleaned',
                y='MRC',
                color='A Loc State',
                markers=True,
                title=' ',
                labels={'Term_Cleaned': 'Contract Term (Months)', 'MRC': 'Average MRC ($)', 'A Loc State': 'State/Country'}
            )

            fig.update_traces(connectgaps=True)
            fig.update_layout(
                xaxis=dict(
                    title='Contract Term (Months)',
                    tickmode='array',
                    tickvals=[12, 24, 36, 48, 60],
                    ticktext=['12', '24', '36', '48', '60'],
                    type='linear'
                ),
                yaxis_title='Average MRC ($)'
            )
            st.plotly_chart(fig, use_container_width=True)







    # Plot 3

    unique_cir_types = filter_state_data['generalized_Cir'].dropna().unique()
    unique_cir_types = sorted(unique_cir_types)
    st.header('Product wise comparison')
    col1, col2 = st.columns([1, 4]) 
    
    with col1:
        st.subheader('Products')
        selected_cir_types = []
        for cir in unique_cir_types:
            if st.checkbox(cir, value=True, key=cir):
                selected_cir_types.append(cir)

    with col2:
        filtered_cir_data = filter_state_data[filter_state_data['generalized_Cir'].isin(selected_cir_types)]
        
        cir_mrc = filtered_cir_data.groupby('generalized_Cir')['MRC'].mean().reset_index()
        
        fig_cir = px.bar(
            cir_mrc,
            x='generalized_Cir',
            y='MRC',
            color='generalized_Cir',
            title=' ',
            labels={'generalized_Cir': 'Circuit Type', 'MRC': 'Average MRC ($)'}
        )

        fig_cir.update_layout(
            xaxis_title='Products',
            yaxis_title='Average MRC ($)',
            showlegend=False
        )

        st.plotly_chart(fig_cir, use_container_width=True)



    with st.container():

        st.header('State-wise Cost Comparison in the US')
        # Plot 4
        col1, spacer, col2 = st.columns([9, 1,  10]) 

        with col1:
            state_name_to_code = {
                "New Jersey": "NJ", "Washington DC": "DC", "Virginia": "VA", "Kansas": "KS", "Texas": "TX",
                "Georgia": "GA", "Illinois": "IL", "Minnesota": "MN", "New York": "NY", "Colorado": "CO",
                "Connecticut": "CT", "Ohio": "OH", "California": "CA", "Washington": "WA", "Kentucky": "KY",
                "Maryland": "MD", "South Carolina": "SC", "Utah": "UT", "Pennsylvania": "PA", "Florida": "FL",
                "Massachusetts": "MA", "Oklahoma": "OK", "Michigan": "MI", "Tennessee": "TN", 
                "North Carolina": "NC", "Wisconsin": "WI", "Arizona": "AZ", "Oregon": "OR", "Missouri": "MO"
            }

            filter_state_data['A Loc State'] = filter_state_data['A Loc State'].str.strip()

            us_states = list(state_name_to_code.keys())

            filtered_us_data = filter_state_data[filter_state_data['A Loc State'].isin(us_states)]

            grouped_avg = filtered_us_data.groupby('A Loc State')['MRC'].mean().reset_index()

            # Map state names to state codes
            grouped_avg['state_code'] = grouped_avg['A Loc State'].map(state_name_to_code)


            us_heatmap_data = {
                'state': grouped_avg['state_code'].tolist(),
                'MRC': grouped_avg['MRC'].round(2).tolist()
            }

            us_df = pd.DataFrame(us_heatmap_data)

            fig_heatmap = px.choropleth(
                us_df,
                locations='state',  
                locationmode="USA-states",
                color='MRC',
                color_continuous_scale="Blues",
                scope="usa",
                hover_name='state',
            )

            fig_heatmap.update_layout(
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    center={"lat": 37.0902, "lon": -95.7129},
                    projection_scale=0.95
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0)
            )

            code_to_state_name = {v: k for k, v in state_name_to_code.items()}
            clicked_points = plotly_events(fig_heatmap, click_event=True, hover_event=False)



        with col2:
            if clicked_points:
                clicked_index = clicked_points[0]['pointIndex']
                clicked_state_code = us_df.iloc[clicked_index]['state']

                clicked_state_full_name = code_to_state_name.get(clicked_state_code)

                st.success(f"🗺️ You clicked on: {clicked_state_full_name}")

                state_data = filtered_us_data[filtered_us_data['A Loc State'] == clicked_state_full_name]

                if not state_data.empty:
                    provider_mrc = state_data.groupby('Provider')['MRC'].mean().reset_index()
                    provider_mrc = provider_mrc.sort_values(by='MRC', ascending=False)

                    fig_bar = px.bar(
                        provider_mrc,
                        x='Provider',
                        y='MRC',

                        title=f"Average MRC per Provider in {clicked_state_full_name}",
                        labels={'MRC': 'Average MRC', 'Provider': 'Provider'},
                        color='MRC',
                        color_continuous_scale='Viridis'
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning(f"No provider data available for {clicked_state_full_name}")







else:
    st.info("👆 Upload a CSV file to begin analysis.")
