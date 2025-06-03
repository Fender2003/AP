import streamlit as st
import base64

def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def render_header():
    logo_base64 = get_base64_of_bin_file("/Users/dhruv/Dhruv/Apcela/download.png")

    st.markdown(
        f"""
        <style>
            /* Hide Streamlit's default menu and padding */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}

            .custom-header {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 80px;
                background-color: #000000;
                z-index: 9999;
                padding: 0 30px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-family: Arial, sans-serif;
                border-bottom: 1px solid #333;
            }}

            .header-left {{
                display: flex;
                align-items: center;
            }}

            .header-logo img {{
                height: 60px;
                margin-right: 30px;
            }}

            .header-menu {{
                display: flex;
                gap: 20px;
                color: white;
                font-size: 18px;
                font-weight: 500;
            }}

            .header-menu div {{
                cursor: pointer;
            }}

            .header-menu div:hover {{
                text-decoration: underline;
            }}

            .header-right {{
                display: flex;
                align-items: center;
                gap: 20px;
            }}

            .contact-button {{
                background-color: #006ac6;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 16px;
                text-decoration: none;
            }}

            .contact-button:hover {{
                background-color: #005bb5;
            }}

            .search-icon {{
                font-size: 22px;
                cursor: pointer;
                color: white;
            }}

            /* Push the page content down */
            .custom-offset {{
                margin-top: 100px;
            }}
        </style>

        <div class="custom-header">
            <div class="header-left">
                <div class="header-logo">
                    <a href="https://www.apcela.com" target="_self">
                        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
                    </a>
                </div>
                <div class="header-menu">
                    <div><a href="https://www.apcela.com/managed-services/" target="_self" style="color:white; text-decoration:none">Services ▾</a></div>
                    <div><a href="https://www.apcela.com/why-apcela/" target="_self" style="color:white; text-decoration:none">Our Platform ▾</a></div>
                    <div><a href="https://www.apcela.com/resources/" target="_self" style="color:white; text-decoration:none">Resources ▾</a></div>
                    <div><a href="https://www.apcela.com/about/" target="_self" style="color:white; text-decoration:none">Company ▾</a></div>
                    <div><a href="https://www.apcela.com/blog/" target="_self" style="color:white; text-decoration:none">Blog</a></div>
                    <div><a href="https://www.apcela.com/support/" target="_self" style="color:white; text-decoration:none">Support</a></div>
                </div>
            </div>
            <div class="header-right">
                <a href="https://www.apcela.com/contact/" target="_self" class="contact-button" style="color:white; text-decoration:none">Contact</a>
                <div class="search-icon">🔍</div>
            </div>
        </div>

        <div class="custom-offset"></div>
        """,
        unsafe_allow_html=True
    )
