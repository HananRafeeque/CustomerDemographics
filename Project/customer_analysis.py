# Customer Demographics and Behavior Visualization

import pandas as pd
import random # Added for dummy data generation
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATASET_PATH = 'customer_demographics.csv' 

VISUALIZATIONS_DIR = 'visualizations'
OUTPUT_DIR = 'output'

import os
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Data Loading and Cleaning ---
def load_and_clean_data(file_path):
    """Loads the dataset, handles missing values, duplicates, and inconsistencies."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}.")
        print("Please download the dataset and update the DATASET_PATH variable.")
        
        print("Creating a dummy DataFrame for demonstration purposes.")
        data = {
            'CustomerID': range(1, 101),
            'Age': [random.randint(18, 70) for _ in range(100)],
            'Gender': [random.choice(['Male', 'Female']) for _ in range(100)],
            'Region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(100)],
            'Spending': [random.uniform(20, 500) for _ in range(100)],
        }
        df = pd.DataFrame(data)
        df.loc[5, 'Age'] = None
        df.loc[10, 'Spending'] = None
        df = pd.concat([df, df.iloc[15:20]], ignore_index=True)

    print("Initial dataset shape:", df.shape)
    print("Initial missing values:\n", df.isnull().sum())

    # Handle missing values (example: fill Age with median, Spending with mean)
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'].fillna(median_age, inplace=True)
        df['Age'] = df['Age'].astype(int) # Ensure Age is integer after fillna

    if 'Spending' in df.columns:
        mean_spending = df['Spending'].mean()
        df['Spending'].fillna(mean_spending, inplace=True)

    # Handle duplicates
    df.drop_duplicates(inplace=True)

    # Handle inconsistencies (example: standardize Gender column)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.capitalize()
        # Add more cleaning steps as needed based on the actual dataset

    print("Cleaned dataset shape:", df.shape)
    print("Missing values after cleaning:\n", df.isnull().sum())
    print("Cleaned dataset info:")
    df.info()
    print("\nFirst 5 rows of cleaned data:\n", df.head())

    # --- Save Cleaned Data ---
    cleaned_data_dir = 'Cleaned data set'
    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)
    cleaned_data_filepath = os.path.join(cleaned_data_dir, 'cleaned_customer_demographics.csv')
    df.to_csv(cleaned_data_filepath, index=False)
    print(f"\nCleaned data saved to: {cleaned_data_filepath}")

    return df

# --- 2. Exploratory Data Analysis (EDA) ---
def perform_eda(df):
    """Performs EDA, calculating summary statistics and correlations."""
    print("\n--- Exploratory Data Analysis ---")

    numerical_cols = df.select_dtypes(include=['number']).columns
    
    if 'Spending' in df.columns:
        df['Spending'] = pd.to_numeric(df['Spending'], errors='coerce')
    numerical_cols = df.select_dtypes(include=['number']).columns # Recalculate after potential type conversion
    print("\nSummary Statistics for Numerical Columns:\n", df[numerical_cols].describe())

    
    correlation_matrix = df[numerical_cols].corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)

    
    if 'Gender' in df.columns and 'Region' in df.columns and 'Spending' in df.columns:
        grouped_data = df.groupby(['Gender', 'Region'])['Spending'].mean().unstack()
        print("\nAverage Spending by Gender and Region:\n", grouped_data)

    # Grouping by Age Group (example)
    if 'Age' in df.columns and 'Spending' in df.columns:
        bins = [18, 30, 45, 60, 100]
        labels = ['18-30', '31-45', '46-60', '60+']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        # Ensure 'Spending' is numeric before aggregation
        df['Spending'] = pd.to_numeric(df['Spending'], errors='coerce')
        age_group_summary = df.groupby('AgeGroup')['Spending'].agg(['mean', 'count'])
        print("\nSpending Summary by Age Group:\n", age_group_summary)

    return df, correlation_matrix

# --- 3. Data Visualization ---
def create_visualizations(df, correlation_matrix):
    """Creates visualizations using Seaborn and Plotly."""
    print("\n--- Creating Visualizations ---")

    # a) Bar chart: Spending by age group (Seaborn)
    if 'AgeGroup' in df.columns and 'Spending' in df.columns:
        plt.figure(figsize=(10, 6))
        
        df['Spending'] = pd.to_numeric(df['Spending'], errors='coerce')
        
        age_group_spending = df.groupby('AgeGroup')['Spending'].sum().reset_index()
        sns.barplot(x='AgeGroup', y='Spending', data=age_group_spending, errorbar=None, palette='viridis')
        plt.title('Total Spending by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Total Spending ($)')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'total_spending_by_age_group.png'))
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'total_spending_by_age_group.png')}")
        # plt.show() # Optional: display plot
        plt.close()

    # b) Interactive Plot: Average Spending by Region and Gender (Scatter Plot)
    if all(col in df.columns for col in ['Region', 'Gender', 'Spending']):
        
        df['Spending'] = pd.to_numeric(df['Spending'], errors='coerce')
        df.dropna(subset=['Spending', 'Region', 'Gender'], inplace=True) 

        avg_spending_df = df.groupby(['Region', 'Gender'])['Spending'].mean().reset_index()

        avg_scatter_fig = px.scatter(
            avg_spending_df,
            x='Region',
            y='Spending',
            color='Gender',
            size='Spending',
            title='Average Spending by Region and Gender',
            labels={'Spending': 'Average Spending ($)', 'Region': 'Region'},
            template='plotly_white'
        )
        avg_scatter_fig.update_layout(legend_title_text='Gender')
        avg_scatter_fig.write_html(os.path.join(VISUALIZATIONS_DIR, 'avg_spending_scatter_by_demographics.html'))
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'avg_spending_scatter_by_demographics.html')}")
        
    if 'Product Category' in df.columns and 'Spending' in df.columns:
        plt.figure(figsize=(12, 6))

        product_spending = df.groupby('Product Category')['Spending'].sum().sort_values(ascending=False).reset_index()
        sns.barplot(x='Spending', y='Product Category', data=product_spending, palette='crest')
        plt.title('Top Product Categories by Total Spending')
        plt.xlabel('Total Spending ($)')
        plt.ylabel('Product Category')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'top_product_categories_total_spending.png'))
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'top_product_categories_total_spending.png')}")
        plt.close()

    

# --- 4. Interactive Dashboard (Plotly) ---
def create_dashboard(df):
    """Creates a more visually rich interactive dashboard using Plotly."""
    print("\n--- Creating Enhanced Interactive Dashboard ---")

    required_cols = ['AgeGroup', 'Gender', 'Region', 'Spending']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Skipping dashboard creation due to missing required columns: {', '.join(missing_cols)}")
        return
    
    df['Spending'] = pd.to_numeric(df['Spending'], errors='coerce')
    df.dropna(subset=['Spending'], inplace=True)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        subplot_titles=(
            "Customer Distribution by Gender",
            "Total Spending by Age Group",
            "Spending Boxplot by Age Group",
            "Total Spending by Region",
        )
    )

    fig.update_layout(template='plotly_white')

    # Pie chart - Gender distribution
    gender_counts = df['Gender'].value_counts()
    fig.add_trace(go.Pie(labels=gender_counts.index, values=gender_counts.values), row=1, col=1)

    # Bar chart - Total spending by AgeGroup
    age_group_spending = df.groupby('AgeGroup')['Spending'].sum().reset_index()
    fig.add_trace(go.Bar(x=age_group_spending['AgeGroup'], y=age_group_spending['Spending'],
                         marker_color='indianred', name='Age Group Spending'), row=1, col=2)

    # Histogram - Spending distribution
    # fig.add_trace(go.Histogram(x=df['Spending'], marker_color='mediumseagreen', nbinsx=20), row=2, col=1)
    # fig.update_xaxes(title_text='Spending Amount', row=2, col=1)
    # fig.update_yaxes(title_text='Number of Customers', row=2, col=1)

    # Bar chart - Total spending by Region
    region_spending = df.groupby('Region')['Spending'].sum().reset_index()
    fig.add_trace(go.Bar(x=region_spending['Region'], y=region_spending['Spending'],
                         marker_color='dodgerblue', name='Region Spending'), row=2, col=2)

    # Boxplot - Spending by AgeGroup
    fig.add_trace(go.Box(x=df['AgeGroup'], y=df['Spending'], name='Spending by AgeGroup',
                         marker_color='mediumpurple'), row=2, col=1)


    fig.update_layout(
        title_text='Enhanced Customer Demographics and Spending Dashboard',
        height=1200,
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    output_path = os.path.join(OUTPUT_DIR, 'enhanced_customer_dashboard.html')
    fig.write_html(output_path)
    print(f"✅ Saved dashboard: {output_path}")


# --- 5. Main Execution --- 
if __name__ == "__main__":
    print("Starting Customer Demographics Analysis...")
    
    # Add a check for dummy data creation
    import random # Make sure random is imported if using dummy data

    # 1. Load and Clean Data
    customer_df = load_and_clean_data(DATASET_PATH)

    # Check if DataFrame is valid before proceeding
    if customer_df is not None and not customer_df.empty:
        # 2. Perform EDA
        customer_df_eda, corr_matrix = perform_eda(customer_df)

        # 3. Create Visualizations
        create_visualizations(customer_df_eda, corr_matrix)

        # 4. Create Dashboard
        create_dashboard(customer_df_eda)


        print("\nAnalysis Complete. Outputs saved in 'visualizations' and 'output' folders.")
    else:
        print("\nAnalysis could not proceed due to data loading issues.")
