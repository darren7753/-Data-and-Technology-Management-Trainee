import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from streamlit_option_menu import option_menu

# Settings
st.set_page_config(
    page_title="Tue 17 Sep 2024 Report",
    layout="wide"
)

# Function
def input_missing_values(df, target_col, group_col, method="mean"):
    """
    Input missing values in a DataFrame based on group-specific statistics.

    This function fills missing values in the specified target column by calculating
    a statistic (mean, median, or mode) within each group of the specified grouping column.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    target_col : str
        The name of the column with missing values to be filled.
    group_col : str
        The name of the column used to group data before calculating the statistic.
    method : str, optional
        The method to fill missing values. Options are 'mean', 'median', or 'mode'.
    
    Raises:
    -------
    ValueError
        If an invalid method is provided (i.e., not 'mean', 'median', 'mode', or None).
    """

    groups = df[group_col].unique()
    
    for group in groups:
        filter = df[group_col] == group
        
        if method == "mean":
            replacement_value = df[filter][target_col].mean()
        elif method == "median":
            replacement_value = df[filter][target_col].median()
        elif method == "mode":
            replacement_value = df[filter][target_col].mode()[0]
        else:
            raise ValueError("Invalid method. Choose 'mean', 'median', 'mode', or leave it as None for default behavior.")

        df.loc[filter, target_col] = df.loc[filter, target_col].fillna(replacement_value)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("../datasets/customer-data.csv")
    return df

df = load_data()
df = df.drop(["id", "DUIs"], axis=1)
input_missing_values(df, target_col="credit_score", group_col="income")
input_missing_values(df, target_col="annual_mileage", group_col="driving_experience")

# Sidebar for navigation
st.html("""
    <style>
        [alt=Logo] {
            height: 4.5rem;
            border-radius: 10px;
            background-color: #FAFAFA;
            padding: 5px;
        }
    </style>
""")
st.logo(
    image="assets/logo_panjang.png",
    link="https://fifgroup.co.id/",
    icon_image="assets/logo_pendek.png"
)

with st.sidebar:
    st.title("Tue 17 Sep 2024 Report")

    option = option_menu(
        "Navigation",
        ["Categorical Univariate", "Numerical Univariate", "Bivariate", "Multivariate"],
        icons=["1-circle-fill", "2-circle-fill", "3-circle-fill", "4-circle-fill"],
        default_index=0
    )

    st.write("Made with ❤️ by **Mathew Darren Kusuma**")

# Content
if option == "Categorical Univariate":
    st.header("Education Univariate Analysis")
    st.write("")
    st.write("")

    df["education"] = pd.Categorical(df["education"], categories=["none", "high school", "university"], ordered=True)
    
    education_counts = df["education"].value_counts().reset_index()
    education_counts.columns = ["education", "count"]

    bar_chart = alt.Chart(education_counts).mark_bar(cornerRadiusTopLeft=25, cornerRadiusTopRight=25).encode(
        x=alt.X("education", sort=["none", "high school", "university"], title="Education Level", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count", title="Count"),
        color=alt.Color("education", legend=None),
        tooltip=[alt.Tooltip("education:N", title="Education Level"), alt.Tooltip("count:Q", title="Count")]
    ).properties(
        height=500
    )

    text = bar_chart.mark_text(
        align='center',
        baseline='middle',
        dy=-20,
        fontSize=20
    ).encode(
        text="count:Q"
    )

    combined_chart = (bar_chart + text).configure_axis(
        labelFontSize=16,
        titleFontSize=18,
        grid=False
    ).configure_view(
        strokeOpacity=0
    )

    st.altair_chart(combined_chart, use_container_width=True)

    st.info("""
        From the plot, it’s clear that the number of high school and university clients is roughly the same, while the 'none' category is significantly smaller by comparison. However, there are a few issues to address:
            
        - The 'none' category is ambiguous. It might include people who aren't currently in school, those who didn't complete their education, or other possibilities. Additionally, there’s no representation for primary or junior high school, and no explanation is given for this omission.
        - The stark contrast between the 'none' category and the high school and university categories is unclear. This discrepancy might be due to data collected from a location where high school and university students are prevalent, random chance, or other factors.
            
        More details about the dataset and its collection process would be helpful.
    """, icon="ℹ️")

elif option == "Numerical Univariate":
    st.header("Annual Mileage Univariate Analysis")
    st.write("")
    st.write("")

    plot_type = st.selectbox("Select the plot type", ["Histogram", "Distribution Plot", "Boxplot"])

    if plot_type == "Histogram":
        chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
            alt.X("annual_mileage:Q", bin=alt.Bin(maxbins=30), title="Annual Mileage"),
            y="count()",
            tooltip=[alt.Tooltip("count()", title="Count")]
        ).properties(
            height=400
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
            grid=False,
        ).configure_view(
            strokeOpacity=0
        )
        st.altair_chart(chart, use_container_width=True)

    elif plot_type == "Distribution Plot":
        kde_data = pd.DataFrame({'annual_mileage': np.linspace(df['annual_mileage'].min(), df['annual_mileage'].max(), 500)})
        kde_data['density'] = np.exp(-0.5 * ((kde_data['annual_mileage'] - df['annual_mileage'].mean()) / df['annual_mileage'].std())**2)
        
        chart = alt.Chart(kde_data).mark_line().mark_area().encode(
            alt.X("annual_mileage:Q", title="Annual Mileage"),
            y=alt.Y("density:Q", title="Density"),
            tooltip=[alt.Tooltip("annual_mileage:Q", title="Mileage"), alt.Tooltip("density:Q", title="Density")]
        ).properties(
            height=400
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
            grid=False,
        ).configure_view(
            strokeOpacity=0
        )
        st.altair_chart(chart, use_container_width=True)

    elif plot_type == "Boxplot":
        chart = alt.Chart(df).mark_boxplot(size=70, color='lightblue', median={'color': 'black'}).encode(
            alt.X("annual_mileage:Q", title="Annual Mileage"),
            tooltip=[alt.Tooltip("annual_mileage:Q", title="Mileage")]
        ).properties(
            height=400
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
            grid=False,
        ).configure_view(
            strokeOpacity=0
        )
        st.altair_chart(chart, use_container_width=True)

    st.dataframe(df[['annual_mileage']].describe().T, use_container_width=True)

    st.info("""
        Visually, the annual mileage column appears to be normally distributed, as evidenced by its symmetrical histogram and bell-shaped curve.
            
        A normal distribution is beneficial because it indicates that the data is spread in a predictable manner, with most values concentrated around the mean and fewer values occurring at the extremes. This distribution is ideal for applying various statistical methods, many of which assume normality. For example, techniques such as regression analysis, hypothesis testing, and confidence intervals rely on the assumption that the data follows a normal distribution.
            
        As the annual mileage column demonstrates a normal distribution, it is well-suited for further analysis using these statistical methods.
    """, icon="ℹ️")

elif option == "Bivariate":
    st.header("Education vs Outcome Bivariate Analysis")
    st.write("")
    st.write("")
    
    df_grouped = df.groupby(['education', 'outcome']).size().reset_index(name='count')
    
    df_grouped['total'] = df_grouped.groupby('education')['count'].transform('sum')
    
    df_grouped['percentage'] = df_grouped['count'] / df_grouped['total']
    
    stacked_chart = alt.Chart(df_grouped).mark_bar(cornerRadiusTopLeft=25, cornerRadiusTopRight=25).encode(
        x=alt.X('education:N', sort=["none", "high school", "university"], title="Education Level", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('percentage:Q', axis=alt.Axis(format='%'), title="Percentage"),
        color=alt.Color('outcome:N', legend=alt.Legend(title="Outcome")),
        tooltip=[
            alt.Tooltip('education:N', title='Education Level'), 
            alt.Tooltip('outcome:N', title='Outcome'), 
            alt.Tooltip('percentage:Q', format='.2%', title='Percentage')
        ]
    ).properties(
        height=500
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=18,
        grid=False
    ).configure_view(
        strokeOpacity=0
    )

    st.altair_chart(stacked_chart, use_container_width=True)

    st.info("""
        The plot illustrates the percentage of insurance claims for each education level, with darker blue representing claimed and lighter blue representing unclaimed. It seems that higher education levels correspond with a lower percentage of claims. This trend deserves further examination.
            
        According to the initial plot, most of our clients are from high school and university backgrounds, but these groups have the lowest claim percentages. The reason for this discrepancy is not immediately clear.
            
        Additionally, the 'none' category is ambiguous and has the lowest number of clients, yet it shows the highest percentage of claims. This discrepancy emphasizes the need to break down the 'none' group further to understand its composition. It is crucial to clarify what 'none' represents, as this category might reveal important insights that are currently obscured.
            
        We should reassess our marketing strategy and identify our target audience more precisely to avoid potentially losing valuable clients. Further analysis of the 'none' category will help us make more informed decisions.
    """, icon="ℹ️")

elif option == "Multivariate":
    st.header("Driving Experience & Marital Status vs Claim Rate Multivariate Analysis")
    st.write("")
    st.write("")
    
    pivot_table = df.pivot_table(
        values='outcome',
        index='driving_experience',
        columns='married',
        aggfunc='mean'
    ).reset_index()

    pivot_table.columns = ['driving_experience', 'Not Married', 'Married']

    df_melted = pivot_table.melt(id_vars='driving_experience', var_name='marital_status', value_name='claim_rate')

    heatmap = alt.Chart(df_melted).mark_rect().encode(
        x=alt.X('marital_status:N', title="Marital Status"),
        y=alt.Y('driving_experience:N', title="Driving Experience"),
        color=alt.Color('claim_rate:Q', scale=alt.Scale(scheme='blues'), title='Claim Rate'),
        tooltip=[
            alt.Tooltip('driving_experience:N', title='Driving Experience'),
            alt.Tooltip('marital_status:N', title='Marital Status'),
            alt.Tooltip('claim_rate:Q', format='.2%', title='Claim Rate')
        ]
    )

    text = alt.Chart(df_melted).mark_text(align='center', baseline='middle', fontSize=24, dx=0, dy=0).encode(
        x=alt.X('marital_status:N'),
        y=alt.Y('driving_experience:N'),
        text=alt.Text('claim_rate:Q', format='.2%'),
        color=alt.value('black')
    )

    combined_chart = (heatmap + text).properties(
        height=500,
        width=500
    )

    combined_chart = combined_chart.configure_axis(
        labelFontSize=16,
        titleFontSize=18
    ).configure_view(
        strokeOpacity=0
    )

    st.altair_chart(combined_chart, use_container_width=True)

    st.info("""
        The plot shows that as driving experience decreases, the claim rate increases. This suggests that less experienced drivers tend to file more insurance claims. Furthermore, within each category of driving experience, individuals who are not married consistently exhibit higher claim rates compared to those who are married.
            
        This pattern implies that driving experience is a significant factor influencing claim rates, with less experienced drivers being more prone to accidents and, consequently, more likely to make claims. The consistently higher claim rates among unmarried individuals could indicate several underlying factors. For instance, unmarried drivers might have different driving behaviors or risk profiles compared to married drivers. They may engage in riskier driving practices or have different lifestyle-related factors that impact their driving.
            
        Additionally, the higher claim rates among unmarried drivers might be influenced by factors such as social or economic conditions that affect their driving patterns. It could also reflect differences in the way married and unmarried individuals approach vehicle ownership and insurance.
    """, icon="ℹ️")