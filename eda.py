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
    df = pd.read_csv("datasets/customer-data.csv")
    return df

df = load_data()
df = df.drop(["id", "DUIs"], axis=1)

input_missing_values(df, target_col="credit_score", group_col="income")
input_missing_values(df, target_col="annual_mileage", group_col="driving_experience")

bool_columns = df.select_dtypes(include='bool').columns
for col in bool_columns:
    df[col] = df[col].replace({True: "True", False: "False"})

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

if option == "Categorical Univariate":
    st.header("Categorical Univariate Analysis")
    st.write("")
    st.write("")

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Custom sorting orders
    sort_orders = {
        "education": ["none", "high school", "university"],
        "age": ["16-25", "26-39", "40-64", "65+"],
        "driving_experience": ["0-9y", "10-19y", "20-29y", "30y+"],
        "income": ["poverty", "working class", "middle class", "upper class"],
        "vehicle_year": ["before 2015", "after 2015"]
    }

    selected_column = st.selectbox(
        "Select a categorical column for analysis (Education is the default view)",
        ["education"] + [col for col in categorical_columns if col != "education"],
        index=0
    )

    st.write("If you're interested in analyzing other categorical variables, you can select them from the dropdown above.")

    def create_categorical_chart(data, column):
        column_counts = data[column].value_counts().reset_index()
        column_counts.columns = [column, "count"]

        sort_order = sort_orders.get(column)
        if sort_order:
            column_counts[column] = pd.Categorical(column_counts[column], categories=sort_order, ordered=True)
            column_counts = column_counts.sort_values(column)
        else:
            sort_order = column_counts[column].tolist()

        bar_chart = alt.Chart(column_counts).mark_bar(cornerRadiusTopLeft=25, cornerRadiusTopRight=25).encode(
            x=alt.X(f"{column}:N", sort=sort_order, title=column.replace('_', ' ').capitalize(), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color(f"{column}:N", legend=None, scale=alt.Scale(scheme='blues')),
            tooltip=[alt.Tooltip(f"{column}:N", title=column.replace('_', ' ').capitalize()), alt.Tooltip("count:Q", title="Count")]
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

        return combined_chart

    chart = create_categorical_chart(df, selected_column)
    st.altair_chart(chart, use_container_width=True)

    if selected_column == "education":
        st.info("""
            From the plot, it's clear that the number of high school and university clients is roughly the same, while the 'none' category is significantly smaller by comparison. However, there are a few issues to address:
                
            - The 'none' category is ambiguous. It might include people who aren't currently in school, those who didn't complete their education, or other possibilities. Additionally, there's no representation for primary or junior high school, and no explanation is given for this omission.
            - The stark contrast between the 'none' category and the high school and university categories is unclear. This discrepancy might be due to data collected from a location where high school and university students are prevalent, random chance, or other factors.
                
            More details about the dataset and its collection process would be helpful.
        """, icon="ℹ️")
    else:
        st.info(f"""
            This chart shows the distribution of the '{selected_column.replace('_', ' ')}' variable in the dataset. 
            You can observe the frequency of each category within this variable.
            
            For a more detailed analysis of this variable, consider:
            1. The overall distribution and any notable patterns
            2. The presence of any outliers or unusual categories
            3. How this variable might relate to other variables in the dataset
            4. Any potential implications for your analysis or business decisions
        """, icon="ℹ️")

elif option == "Numerical Univariate":
    st.header("Numerical Univariate Analysis")
    st.write("")
    st.write("")

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        selected_column = st.selectbox(
            "Select a numerical column for analysis (Annual Mileage is the default view)",
            ["annual_mileage"] + [col for col in numerical_columns if col != "annual_mileage"],
            index=0
        )

    with col2:
        plot_type = st.selectbox("Select the plot type", ["Histogram", "Distribution Plot", "Boxplot"])

    st.write("If you're interested in analyzing other numerical variables, you can select them from the dropdown above.")

    if plot_type == "Histogram":
        chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
            alt.X(f"{selected_column}:Q", bin=alt.Bin(maxbins=30), title=selected_column.replace('_', ' ').title()),
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
        kde_data = pd.DataFrame({selected_column: np.linspace(df[selected_column].min(), df[selected_column].max(), 500)})
        kde_data['density'] = np.exp(-0.5 * ((kde_data[selected_column] - df[selected_column].mean()) / df[selected_column].std())**2)
        
        chart = alt.Chart(kde_data).mark_area().encode(
            alt.X(f"{selected_column}:Q", title=selected_column.replace('_', ' ').title()),
            y=alt.Y("density:Q", title="Density"),
            tooltip=[alt.Tooltip(f"{selected_column}:Q", title=selected_column.replace('_', ' ').title()), alt.Tooltip("density:Q", title="Density")]
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
            alt.X(f"{selected_column}:Q", title=selected_column.replace('_', ' ').title()),
            tooltip=[alt.Tooltip(f"{selected_column}:Q", title=selected_column.replace('_', ' ').title())]
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

    st.dataframe(df[[selected_column]].describe().T, use_container_width=True)

    if selected_column == "annual_mileage":
        st.info("""
            Visually, the annual mileage column appears to be normally distributed, as evidenced by its symmetrical histogram and bell-shaped curve.
                
            A normal distribution is beneficial because it indicates that the data is spread in a predictable manner, with most values concentrated around the mean and fewer values occurring at the extremes. This distribution is ideal for applying various statistical methods, many of which assume normality. For example, techniques such as regression analysis, hypothesis testing, and confidence intervals rely on the assumption that the data follows a normal distribution.
                
            As the annual mileage column demonstrates a normal distribution, it is well-suited for further analysis using these statistical methods.
        """, icon="ℹ️")
    else:
        st.info(f"""
            This analysis shows the distribution of the '{selected_column}' variable in the dataset. 
            
            Key points to consider:
            1. Shape of the distribution: Is it normal, skewed, or multi-modal?
            2. Central tendency: Where is the center of the data (mean, median)?
            3. Spread: How spread out are the values (standard deviation, range)?
            4. Outliers: Are there any unusual values that stand out?
            
            Consider how this distribution might impact your analysis or business decisions, and how it might relate to other variables in the dataset.
        """, icon="ℹ️")

elif option == "Bivariate":
    st.header("Bivariate Analysis")
    st.write("")
    st.write("")

    all_columns = df.columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Select first variable", all_columns, index=all_columns.index('education'))
    with col2:
        var2 = st.selectbox("Select second variable", [col for col in all_columns if col != var1], index=all_columns.index('outcome') - 1)

    var1_type = "categorical" if var1 in categorical_columns else "numerical"
    var2_type = "categorical" if var2 in categorical_columns else "numerical"

    # Custom sorting orders
    sort_orders = {
        "education": ["none", "high school", "university"],
        "age": ["16-25", "26-39", "40-64", "65+"],
        "driving_experience": ["0-9y", "10-19y", "20-29y", "30y+"],
        "income": ["poverty", "working class", "middle class", "upper class"],
        "vehicle_year": ["before 2015", "after 2015"]
    }

    if var1_type == "categorical" and var2_type == "categorical":
        df_grouped = df.groupby([var1, var2]).size().reset_index(name='count')
        df_grouped['total'] = df_grouped.groupby(var1)['count'].transform('sum')
        df_grouped['percentage'] = df_grouped['count'] / df_grouped['total']
        
        sort_order = sort_orders.get(var1)
        
        color_scheme = alt.Scale(scheme='blues')
        
        stacked_chart = alt.Chart(df_grouped).mark_bar(cornerRadiusTopLeft=25, cornerRadiusTopRight=25).encode(
            x=alt.X(f'{var1}:N', sort=sort_order, title=var1.replace('_', ' ').title(), axis=alt.Axis(labelAngle=0)),
            y=alt.Y('percentage:Q', axis=alt.Axis(format='%'), title="Percentage"),
            color=alt.Color(f'{var2}:N', scale=color_scheme, legend=alt.Legend(title=var2.replace('_', ' ').title())),
            tooltip=[
                alt.Tooltip(f'{var1}:N', title=var1.replace('_', ' ').title()), 
                alt.Tooltip(f'{var2}:N', title=var2.replace('_', ' ').title()), 
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

    elif var1_type == "numerical" and var2_type == "numerical":
        scatter_chart = alt.Chart(df).mark_circle().encode(
            x=alt.X(f'{var1}:Q', title=var1.replace('_', ' ').title()),
            y=alt.Y(f'{var2}:Q', title=var2.replace('_', ' ').title()),
            tooltip=[alt.Tooltip(f'{var1}:Q', title=var1.replace('_', ' ').title()),
                     alt.Tooltip(f'{var2}:Q', title=var2.replace('_', ' ').title())]
        ).properties(
            height=500
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
            grid=False
        ).configure_view(
            strokeOpacity=0
        )
        
        st.altair_chart(scatter_chart, use_container_width=True)

    else:
        if var1_type == "categorical":
            cat_var, num_var = var1, var2
        else:
            cat_var, num_var = var2, var1
        
        sort_order = sort_orders.get(cat_var)
        
        chart = alt.Chart(df).mark_boxplot(size=50).encode(
            x=alt.X(f'{cat_var}:N', title=cat_var.replace('_', ' ').title(), sort=sort_order),
            y=alt.Y(f'{num_var}:Q', title=num_var.replace('_', ' ').title()),
            color=alt.Color(f'{cat_var}:N', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title=cat_var.replace('_', ' ').title())),
            tooltip=[alt.Tooltip(f'{cat_var}:N', title=cat_var.replace('_', ' ').title()),
                     alt.Tooltip(f'{num_var}:Q', title=num_var.replace('_', ' ').title())]
        ).properties(
            height=500
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
            grid=False
        ).configure_view(
            strokeOpacity=0
        )
        
        st.altair_chart(chart, use_container_width=True)

    if var1 == "education" and var2 == "outcome":
        st.info("""
            The plot illustrates the percentage of insurance claims for each education level, with orange representing claimed (true) and blue representing unclaimed (false). It seems that higher education levels correspond with a lower percentage of claims. This trend deserves further examination.
                
            According to the plot, most of our clients are from high school and university backgrounds, but these groups have the lowest claim percentages. The reason for this discrepancy is not immediately clear.
                
            Additionally, the 'none' category is ambiguous and has the lowest number of clients, yet it shows the highest percentage of claims. This discrepancy emphasizes the need to break down the 'none' group further to understand its composition. It is crucial to clarify what 'none' represents, as this category might reveal important insights that are currently obscured.
                
            We should reassess our marketing strategy and identify our target audience more precisely to avoid potentially losing valuable clients. Further analysis of the 'none' category will help us make more informed decisions.
        """, icon="ℹ️")
    else:
        st.info(f"""
            This plot shows the relationship between {var1.replace('_', ' ')} and {var2.replace('_', ' ')}.
            
            Key points to consider:
            1. Look for any clear patterns or trends in the data.
            2. Consider the strength and direction of any relationship you observe.
            3. Think about how this relationship might impact your analysis or business decisions.
            4. Consider if there might be any confounding variables affecting this relationship.
            
            Remember that correlation does not imply causation. Further analysis may be needed to understand the nature of any observed relationship.
        """, icon="ℹ️")

elif option == "Multivariate":
    for col in bool_columns:
        df[col] = df[col].replace({"True": True, "False": False})

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