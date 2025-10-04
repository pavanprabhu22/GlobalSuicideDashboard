import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
st.cache_data
def load_data():
    data = pd.read_csv("master.csv")
    return data

data = load_data()

# Validate and clean column names
data.columns = data.columns.str.strip()

# Clean and preprocess 'gdp_for_year ($)' and 'population'
if 'gdp_for_year ($)' in data.columns:
    data['gdp_for_year ($)'] = data['gdp_for_year ($)'].str.replace(',', '').astype(float)

if 'population' in data.columns:
    data['population'] = pd.to_numeric(data['population'], errors='coerce')

# Streamlit app
st.title("Comprehensive Suicide Data Analysis Dashboard")

# Tabs for different perspectives
tabs = st.tabs([
    "Overview", "Country Analysis", "Age Analysis", "Gender Analysis","GDP and Population Analysis", "Correlation Analysis", "Custom Insights","Generation Analysis","Perdictive Insights"
])

# Display raw data in Overview tab
with tabs[0]:
    st.header("Dataset Overview")
    if st.checkbox("Show raw data"):
        st.write(data)

    st.subheader("Basic Statistics")
    st.write(data.describe())
    st.markdown("This dataset contains global suicide statistics from 1985 to 2016, categorised by country, year, sex, age group, and generation. It includes demographic details, suicide numbers, population sizes, and economic indicator like GDP. The data enables analysis of trends, demographic patterns, and correlations between socioeconomic factors and suicide rates.")

# Country Analysis
with tabs[1]:
    st.header("Country-wise Analysis")

    countries = data['country'].unique()
    selected_country = st.multiselect("Select Country", countries, default=countries, key="country_multiselect_tab1")
    data_country = data[data['country'].isin(selected_country)]

    st.subheader("Total Suicides by Country")
    suicides_by_country = data_country.groupby('country')['suicides_no'].sum().sort_values(ascending=False)
    st.bar_chart(suicides_by_country)

    st.subheader("Yearly Trends by Country")
    for country in selected_country:
        st.write(f"Yearly Suicide Trends for {country}")
        country_data = data_country[data_country['country'] == country]
        suicides_by_year = country_data.groupby('year')['suicides_no'].sum()
        st.bar_chart(suicides_by_year)
    st.markdown("This section provides an in-depth analysis of suicides by country. You can select one or more countries to visualize the following:\n- **Total Suicides by Country**: A bar chart showing the total number of suicides in each country based on the selected countries.\n- **Yearly Suicide Trends**: A line chart that illustrates the yearly suicide trends for each selected country, enabling a comparison of changes over time.\nThe data is grouped by country and year, offering insights into the suicide rates across different regions. Adjust the country selection to focus on specific regions or countries of interest.")

# Age Analysis
with tabs[2]:
    st.header("Age Group Analysis")

    st.subheader("Total Suicides by Age Group")
    suicides_by_age = data.groupby('age')['suicides_no'].sum()
    fig, ax = plt.subplots()
    sns.barplot(x=suicides_by_age.index, y=suicides_by_age.values, ax=ax)
    ax.set_title("Suicides by Age Group")
    st.pyplot(fig)

    st.subheader("Yearly Trends by Age Group")
    age_groups = data['age'].unique()
    for age_group in age_groups:
        st.write(f"Yearly Trends for {age_group}")
        age_data = data[data['age'] == age_group]
        suicides_by_year = age_data.groupby('year')['suicides_no'].sum()
        st.line_chart(suicides_by_year)
    st.markdown("This section provides insights into suicide trends based on different age groups. You can explore the following:\n- **Total Suicides by Age Group**: A bar chart displaying the total number of suicides for each age group, providing a clear comparison across different age categories.\n- **Yearly Suicide Trends by Age Group**: A line chart showing the suicide trends over time for each age group, allowing you to see how suicide rates have changed year by year within specific age categories.\nThe analysis helps identify which age groups have been most affected by suicides, as well as any significant trends or fluctuations over time.")

# Gender Analysis
with tabs[3]:
    st.header("Gender-wise Analysis")

    genders = data['sex'].unique()
    selected_gender = st.multiselect("Select Gender", genders, default=genders, key="gender_multiselect_tab3")
    data_gender = data[data['sex'].isin(selected_gender)]

    st.subheader("Total Suicides by Gender")
    suicides_by_gender = data_gender.groupby('sex')['suicides_no'].sum()
    fig, ax = plt.subplots()
    sns.barplot(x=suicides_by_gender.index, y=suicides_by_gender.values, ax=ax)
    ax.set_title("Suicides by Gender")
    st.pyplot(fig)
    st.markdown("This section offers a gender-based analysis of suicides. You can explore the following:\n- **Total Suicides by Gender**: A bar chart illustrating the total number of suicides by gender.\n- **Yearly Suicide Trends by Gender**: A line chart displaying the trends in suicides over the years, segmented by gender.")

with tabs[4]:
    st.header("GDP and Population Analysis")

    st.subheader("GDP vs Suicides")
    if 'gdp_for_year ($)' in data.columns:
        import plotly.express as px

        # GDP vs Suicides: Scatterplot with hover
        fig = px.scatter(
            data,
            x='gdp_for_year ($)',
            y='suicides_no',
            color='country',
            hover_data={
                'country': True,
                'gdp_for_year ($)': ':,.0f',
                'suicides_no': ':,.0f'
            },
            labels={
                'gdp_for_year ($)': 'GDP for Year ($)',
                'suicides_no': 'Number of Suicides'
            },
            title="GDP vs Suicides"
        )

        fig.update_traces(marker=dict(size=10, opacity=0.6), selector=dict(mode='markers'))
        st.plotly_chart(fig)

    else:
        st.error("Column 'gdp_for_year ($)' not found in dataset.")

    st.subheader("Population vs Suicides")

    st.subheader("GDP and Population Correlation")
    if 'gdp_for_year ($)' in data.columns and 'population' in data.columns:
        # Correlation Heatmap using Plotly
        corr_matrix = data[['gdp_for_year ($)', 'population', 'suicides_no']].corr()
        import plotly.figure_factory as ff

        heatmap = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=['GDP for Year ($)', 'Population', 'Suicides'],
            y=['GDP for Year ($)', 'Population', 'Suicides'],
            annotation_text=corr_matrix.round(2).values,
            colorscale="Viridis"
        )

        heatmap.update_layout(title="Correlation Matrix (GDP, Population, Suicides)")
        st.plotly_chart(heatmap)
    else:
        st.error("Required columns for correlation analysis are missing.")

    st.markdown(
        "Here, we analyze the relationship between GDP, population, and suicide rates:\n"
        "- **GDP vs Suicides**: Scatterplot with a regression trendline and hover annotations.\n"
        "- **Population vs Suicides**: Scatterplot with logarithmic scaling and hover annotations.\n"
        "- **GDP and Population Correlation**: Heatmap showing correlations between GDP, population, and suicide numbers."
    )

# Correlation Analysis
with tabs[5]:
    st.header("Correlation Analysis")

    correlation = data.select_dtypes(include=['float64', 'int64']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.markdown("This section provides an overview of the correlation between different numerical variables:\n- **Correlation Heatmap**: A heatmap displaying the correlations between various numerical columns, helping identify relationships between variables like GDP, population, and suicide rates.")

# Custom Insights
import plotly.graph_objects as go

with tabs[6]:
    st.header("Custom Insights")
    st.markdown("Explore advanced insights using the filters provided in the sidebar!")

    # Sidebar filters
    # Sidebar filters with checkboxes
    st.sidebar.header("Filters")
    selected_country = [country for country in countries if st.sidebar.checkbox(f"Select {country}", value=True)]

    selected_year = st.sidebar.slider("Select Year Range", int(data['year'].min()), int(data['year'].max()), (int(data['year'].min()), int(data['year'].max())))
    selected_gender = st.sidebar.multiselect("Select Gender", genders, default=genders, key="sidebar_gender_multiselect")

    # Filter data based on sidebar inputs
    filtered_data = data[data['country'].isin(selected_country)]
    filtered_data = filtered_data[(filtered_data['year'] >= selected_year[0]) & (filtered_data['year'] <= selected_year[1])]
    filtered_data = filtered_data[filtered_data['sex'].isin(selected_gender)]

    st.write("Filtered Dataset")
    st.write(filtered_data)

    # Interactive Pie Chart: Contribution of each country to total suicides
    st.subheader("Country's Contribution to Total Suicides")
    country_contribution = filtered_data.groupby('country')['suicides_no'].sum().reset_index()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=country_contribution['country'],
                values=country_contribution['suicides_no'],
                hoverinfo='label+percent',
                textinfo='none',
                marker=dict(colors=px.colors.sequential.Peach)
            )
        ]
    )
    fig.update_layout(title="Contribution by Country")
    st.plotly_chart(fig)

    

    st.markdown("""
    - **Interactive Pie Chart**: Visualizes each country's contribution to total suicides in the filtered data with hovering enabled.
    - **Custom Total Suicides by Country**: Bar chart showing total suicides by country based on filters.
    - **Custom Yearly Trends**: Line chart displaying yearly suicide trends in the filtered dataset.
    - **Custom Suicides by Age Group**: Bar chart visualizing suicides grouped by age, based on selected filters.
    """)





# Generation Analysis
with tabs[7]:  
    st.header("Generation-wise Analysis")

    st.subheader("Total Suicides by Generation")
    suicides_by_generation = data.groupby('generation')['suicides_no'].sum()
    fig, ax = plt.subplots()
    sns.barplot(x=suicides_by_generation.index, y=suicides_by_generation.values, ax=ax)
    ax.set_title("Total Suicides by Generation")
    ax.set_ylabel("Suicides Count")
    st.pyplot(fig)

    st.subheader("Yearly Trends by Generation")
    generations = data['generation'].unique()
    for generation in generations:
        st.write(f"Yearly Trends for {generation}")
        generation_data = data[data['generation'] == generation]
        suicides_by_year = generation_data.groupby('year')['suicides_no'].sum()
        st.line_chart(suicides_by_year)
    st.markdown("This section focuses on the analysis of suicides across different generational groups:\n- **Total Suicides by Generation**: A bar chart representing the total suicides by generation.\n- **Yearly Suicide Trends by Generation**: A line chart showing the trends in suicides over time for each generation, allowing for an in-depth analysis of generational differences.")


with tabs[8]:  # Add a new tab for Linear Regression
    st.header("Suicide Prediction Using Linear Regression")
    st.markdown("This section predicts future suicides based on historical data using linear regression.")

    # Filtered data preparation
    grouped_data = data.groupby('year')['suicides_no'].sum().reset_index()

    # Prepare features and target
    X = grouped_data[['year']]  # Feature: Year
    y = grouped_data['suicides_no']  # Target: Suicides

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict for test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"### Model Performance Metrics")
    st.markdown(f"- **Mean Squared Error (MSE):** {mse:.2f}")
    st.markdown(f"- **R-squared (R²):** {r2:.2f}")

    # Predict for entire range
    grouped_data['predicted_suicides'] = model.predict(X)

    # Future predictions
    future_years = np.arange(grouped_data['year'].max() + 1, grouped_data['year'].max() + 6).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Display future predictions
    st.subheader("Future Suicide Predictions")
    future_data = pd.DataFrame({'year': future_years.flatten(), 'predicted_suicides': future_predictions})
    st.write(future_data)

    # Plot historical and future predictions
    st.subheader("Historical and Future Predictions")
    fig, ax = plt.subplots()
    ax.plot(grouped_data['year'], grouped_data['suicides_no'], label="Actual Suicides", marker="o", color="blue")
    ax.plot(grouped_data['year'], grouped_data['predicted_suicides'], label="Predicted Suicides", linestyle="--", color="red")
    ax.plot(future_data['year'], future_data['predicted_suicides'], label="Future Predictions", linestyle="-.", color="green")
    ax.set_title("Historical and Future Predictions")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Suicides")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    - **Future Predictions:** Table and graph displaying predicted suicide counts for the next five years.
    - **Combined Visualization:** A comprehensive graph that includes actual suicides, model predictions for historical data, and predictions for future years.
    """)

