import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime

st.set_page_config(page_title="Car Price Analyzer", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('public.csv')
        # Standard cleaning
        df = df.replace(r'\xa0', '', regex=True)
        df = df.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Numeric conversions
        df['price_num'] = pd.to_numeric(df['price'].replace(r'\D', '', regex=True), errors='coerce')
        df['mileage_num'] = pd.to_numeric(df['mileage'].replace(r'\D', '', regex=True), errors='coerce')
        df['year_num'] = pd.to_numeric(df['year'], errors='coerce')

        # Drop rows where we couldn't get a price or mileage
        df = df.dropna(subset=['price_num', 'mileage_num', 'year_num'])

        # --- AGE CALCULATION ---
        current_year = datetime.now().year
        df['age'] = current_year - df['year_num']

        # Sorting Logic
        df = df.sort_values(by='year_num', ascending=True)
        df['year'] = df['year_num'].astype(str).str.replace('.0', '', regex=False)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None

def calculate_if_good_deal(filtered_df, full_df, selected_title):
    st.subheader(f"Analysis & Price Predictor for: {selected_title}")
    
    # Ensure we use the freshly filtered data for the specific model
    model_df = filtered_df.dropna(subset=['price_num', 'age', 'mileage_num'])

    if len(model_df) > 3:
        # 1. Regression (Price ~ Age + Mileage)
        X = model_df[['age', 'mileage_num']]
        X = sm.add_constant(X)
        y = model_df['price_num']

        model = sm.OLS(y, X).fit()
        
        # Extract coefficients
        b0 = model.params['const']           # Base Price (New car)
        b_age = model.params['age']          # Value lost per year
        b_mileage = model.params['mileage_num'] # Value lost per mileage unit

        # 2. UI for the Prediction Tool
        st.write("---")
        st.markdown("### 🔮 Predict a Fair Price")
        col1, col2 = st.columns(2)
        
        current_year = datetime.now().year
        with col1:
            input_year = st.number_input("Enter Year", min_value=1990, max_value=current_year + 1, value=int(model_df['year_num'].mean()))
            # Distance from current year
            calc_age = current_year - input_year
        with col2:
            input_mileage = st.number_input("Enter Mileage (mil)", min_value=0, value=int(model_df['mileage_num'].mean()))

        # Calculate prediction: Base + (Age Coeff * Age) + (Mil Coeff * Mil)
        predicted_price = b0 + (b_age * calc_age) + (b_mileage * input_mileage)
        
        # Floor price at 0
        display_price = max(0, int(predicted_price))
        
        st.metric("Estimated Market Price", f"{display_price:,} kr".replace(',', ' '))
        
        # Updated Formula Caption
        st.caption(f"**Formula:** Price = {int(b0):,} (Base) + ({int(b_age)} * age) + {int(1000*b_mileage)} * Thousand Mil")
        st.write("---")

if df is not None:
    st.title("📈 Car Market Price vs. Mileage")

    title_counts = df['title'].value_counts()
    common_titles = title_counts[title_counts > 1].index.tolist()

    st.sidebar.header("Filter by Common Models")
    for title in common_titles:
        if st.sidebar.button(f"{title} ({title_counts[title]} hits)"):
            st.session_state.selected_title = title

    if st.session_state.selected_title:
        selected_title = st.session_state.selected_title
        st.subheader(f"Analysis for: {selected_title}")
        filtered_df = df[df['title'] == selected_title]        

        fig = px.scatter(
            filtered_df,
            x="mileage_num",
            y="price_num",
            color="year",
            trendline="ols",
            trendline_scope="overall",
            title=f"Market Distribution: {selected_title}",
            labels={"mileage_num": "Mileage (mil)", "price_num": "Price (SEK)"},
            hover_data=['location', 'year', 'mileage', 'price'],
        )
        fig.update_layout(xaxis_title="Mileage (mil)", yaxis_title="Price (kr)")

        avg_price = filtered_df['price_num'].mean()
        st.metric("Average Price", f"{int(avg_price):,} kr".replace(',', ' '))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            filtered_df[['title', 'price', 'year', 'mileage', 'location', 'link']],
            column_config={"link": st.column_config.LinkColumn("Listing Link")},
            use_container_width=True,
            hide_index=True
        )
        calculate_if_good_deal(filtered_df, df, selected_title)
    else:
        st.info("👈 Select a car from the sidebar to see the trendline.")
else:
    st.warning("Data load failed. Check your CSV format.")