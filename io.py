# -*- coding: utf-8 -*-
"""HoT-SEM_Integrated_Analysis_Refactored_V3_Static_Visualizations_Final_v2.ipynb

This version addresses the feedback on the previous version:

- **Arrowheads on SEM Diagrams:** Adds arrowheads to the edges in the SEM
  diagrams using Matplotlib's `arrow` function with appropriate parameters.
- **Black Background for SEM Diagrams:**  Ensures the SEM diagram backgrounds
  are black by setting `facecolor` appropriately.
- **Consistent Dark Graphite Background:**  Ensures *all* parts of *all* plots
  have the dark graphite background, including areas outside the plot area
  itself.  This is achieved by setting both `figure.facecolor` and
  `axes.facecolor` (and using `sns.set_style` correctly).
- **Perfect Circles in SEM Diagrams:** Uses `plt.Circle` to ensure the nodes
  in the SEM diagrams are perfect circles.
- **Removed Redundant `apply_neon_theme_mpl` Calls:** The theme is now applied
  *once* globally.
- **Improved Comments:** Adds more comments for clarity.
"""

# --- Imports ---
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import os
import random
from collections import defaultdict

# --- Install Kaleido (for Plotly image export, if needed in the future) ---
!pip install -U kaleido

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Define Output Directory ---
output_dir = '/content/drive/MyDrive/data'  # Path to the 'data' folder
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

# --- Explicitly set Matplotlib backend to 'Agg' for Colab ---
plt.switch_backend('Agg')

# --- Neon Theme Function (for Matplotlib/Seaborn) ---
def apply_neon_theme_mpl():
    """Applies a dark graphite background and neon colors to Matplotlib plots."""
    plt.style.use('dark_background')  # Set dark background
    plt.rcParams['axes.facecolor'] = '#262626'  # Dark graphite for plot area
    plt.rcParams['figure.facecolor'] = '#262626' # Dark graphite for figure
    plt.rcParams['text.color'] = '#00FF00'  # Bright green text
    plt.rcParams['axes.labelcolor'] = '#00FFFF'  # Cyan axis labels
    plt.rcParams['xtick.color'] = '#00FFFF'
    plt.rcParams['ytick.color'] = '#00FFFF'
    plt.rcParams['grid.color'] = '#444444'  # Darker grid lines
    # For Seaborn, set the style and palette
    sns.set_style("darkgrid", {"axes.facecolor": "#262626", "grid.color": "#444444"}) # Correctly set facecolor for seaborn
    sns.set_palette("bright")  # Use a bright, neon-like palette


# --- Data Simulation (same as V2) ---
def simulate_data(n_participants=40, seed=42):
    """Simulates data, including demographics, interventions, psychological
    measures, performance, and neurophysiological data.  Effects of LLM and
    herbal blend are simulated.

    Args:
        n_participants (int): Number of participants.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: The simulated dataset.
    """

    np.random.seed(seed)

    # Demographics
    age = np.random.randint(18, 30, size=n_participants)
    gender = np.random.choice(['Male', 'Female', 'Other'], size=n_participants)
    programming_experience = np.random.choice(
        ['Beginner', 'Intermediate', 'Advanced'], size=n_participants
    )

    # Group Assignment (balanced)
    llm_usage = np.array([1, 1, 0, 0] * (n_participants // 4))
    herbal_blend = np.array([1, 0, 1, 0] * (n_participants // 4))

    # Psychological Measures (initial and final)
    initial_self_efficacy = np.random.normal(3.5, 0.5, size=n_participants)
    initial_anxiety = np.random.normal(2.5, 0.6, size=n_participants)
    final_self_efficacy = initial_self_efficacy.copy()
    final_anxiety = initial_anxiety.copy()

    # Performance Measures
    errors_identified = np.random.randint(5, 20, size=n_participants)
    completion_time = np.random.uniform(180, 400, size=n_participants)

    # Adjust based on group (simulated effects)
    for i in range(n_participants):
        if llm_usage[i] == 1:
            final_self_efficacy[i] += 0.5
            final_anxiety[i] -= 0.4
            errors_identified[i] += 3
            completion_time[i] -= 15
        if herbal_blend[i] == 1:
            final_anxiety[i] -= 0.3
            errors_identified[i] += 1

    # Ensure reasonable bounds
    final_self_efficacy = np.clip(final_self_efficacy, 1, 5)
    final_anxiety = np.clip(final_anxiety, 1, 4)
    errors_identified = np.maximum(0, errors_identified)
    completion_time = np.maximum(60, completion_time)

    # Neurophysiological Data (simplified)
    eeg_alpha = np.random.normal(10, 2, size=n_participants)
    eeg_beta = np.random.normal(18, 3, size=n_participants)
    ecg_hr = np.random.normal(75, 10, size=n_participants)
    eda_scr = np.random.normal(0.5, 0.2, size=n_participants)
    pog_fixations = np.random.randint(20, 100, size=n_participants)
    pog_fixation_duration = np.random.uniform(200, 500, size=n_participants)
    pog_pupil_diameter = np.random.normal(3.5, 0.5, size=n_participants)
    pog_blink_rate = np.random.uniform(10, 30, size=n_participants)

    # Adjust based on group (simulated effects)
    for i in range(n_participants):
        if llm_usage[i] == 1:
            eeg_beta[i] += 2
            pog_fixations[i] -= 5
            pog_fixation_duration[i] += 50
        if herbal_blend[i] == 1:
            ecg_hr[i] -= 5
            eda_scr[i] -= 0.1

    # Create DataFrame
    data = pd.DataFrame({
        'ParticipantID': range(1, n_participants + 1),
        'Age': age,
        'Gender': gender,
        'ProgrammingExperience': programming_experience,
        'LLMUsage': llm_usage,
        'HerbalBlend': herbal_blend,
        'InitialSelfEfficacy': initial_self_efficacy,
        'FinalSelfEfficacy': final_self_efficacy,
        'InitialAnxiety': initial_anxiety,
        'FinalAnxiety': final_anxiety,
        'ErrorsIdentified': errors_identified,
        'CompletionTime': completion_time,
        'EEGAlpha': eeg_alpha,
        'EEGBeta': eeg_beta,
        'ECG_HR': ecg_hr,
        'EDA_SCR': eda_scr,
        'POGFixations': pog_fixations,
        'POGFixationDuration': pog_fixation_duration,
        'POGPupilDiameter': pog_pupil_diameter,
        'POGBlinkRate': pog_blink_rate
    })

    return data

# --- Data Preprocessing (same as V2) ---

def preprocess_data(data):
    """Preprocesses data: one-hot encodes categoricals, scales numericals,
    and splits into training and testing sets.

    Args:
        data (pd.DataFrame): The raw data.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) preprocessed data splits.
    """

    features = data.drop(columns=['ParticipantID', 'ErrorsIdentified', 'CompletionTime'])
    performance = data[['ErrorsIdentified', 'CompletionTime']]
    features = pd.get_dummies(features, columns=['Gender', 'ProgrammingExperience'])
    numerical_features = features.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    features[numerical_features] = scaler.fit_transform(features[numerical_features])
    X_train, X_test, y_train, y_test = train_test_split(
        features, performance, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# --- Statistical Analyses (same as V2) ---

def perform_statistical_analysis(data):
    """Performs descriptive stats, correlations, and group comparisons (t-tests).

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        tuple: (descriptive_stats, correlation_matrix, group_comparison_results)
    """

    descriptive_stats = data.describe()
    correlation_matrix = data[[
        'FinalSelfEfficacy', 'FinalAnxiety', 'ErrorsIdentified', 'CompletionTime'
    ]].corr()
    group_comparison_results = {}
    for variable in ['FinalSelfEfficacy', 'FinalAnxiety', 'ErrorsIdentified', 'CompletionTime']:
        llm_group = data[data['LLMUsage'] == 1][variable]
        no_llm_group = data[data['LLMUsage'] == 0][variable]
        t_stat, p_val = stats.ttest_ind(llm_group, no_llm_group)
        group_comparison_results[variable] = {'t-statistic': t_stat, 'p-value': p_val}
    return descriptive_stats, correlation_matrix, group_comparison_results


def perform_regression_analysis(X_train, y_train, dependent_variable='ErrorsIdentified'):
    """Performs regression analysis using statsmodels.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target.
        dependent_variable (str): Dependent variable to predict.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Results.
    """

    formula = f"{dependent_variable} ~ LLMUsage + HerbalBlend + InitialSelfEfficacy + InitialAnxiety"
    y, X = dmatrices(formula, data=pd.concat([X_train, y_train], axis=1), return_type='dataframe')
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

# --- Qualitative Analysis (same as V2, with zero-division handling) ---

def analyze_prompts(data):
    """Simulates prompt analysis, generating more realistic prompt data
    based on LLM usage and then analyzing it.  Handles potential
    ZeroDivisionError.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        dict: Analysis results, including generated prompts.
    """
    prompts = []
    for i in range(len(data)):
        if data['LLMUsage'][i] == 1:
            # Simulate more specific prompts for LLM users
            prompt_type = random.choice(["debug", "explain", "optimize"])
            if prompt_type == "debug":
                prompts.append(f"P{i+1}: Find the error in this code: `x = 10; y = 0; z = x / y`")
            elif prompt_type == "explain":
                prompts.append(f"P{i+1}: Explain what this function does: `def add(a, b): return a + b`")
            else:  # optimize
                prompts.append(f"P{i+1}: How can I make this code faster: `for i in range(1000000): pass`")
        else:
            # Simulate more general questions for non-LLM users
            prompts.append(f"P{i+1}: I'm stuck on this task, can you give me a hint?")

    # Analyze the generated prompts
    prompt_lengths = [len(p.split()) for p in prompts]
    # Handle potential ZeroDivisionError if prompt_lengths is empty
    average_prompt_length = np.mean(prompt_lengths) if prompt_lengths else 0

    # Count keywords (more robustly)
    keyword_counts = defaultdict(int)
    for p in prompts:
        for word in p.lower().split():
            if word not in ["i", "this", "the", "a", "in", "on", "can", "you", "me", "what", "how", "is", "do", "does", "an", "here", "fix"]: # Common words
                keyword_counts[word] += 1
    most_common_keywords = sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:5]


    prompt_analysis = {
        "prompts": prompts,  # Include the generated prompts
        "average_prompt_length": average_prompt_length,
        "most_common_keywords": most_common_keywords,
        "question_types": ["debug", "explain", "optimize", "general help"],  # Based on simulation
    }
    return prompt_analysis


def analyze_interviews(data):
    """Simulates interview analysis, generating more realistic feedback.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        dict: Analysis results, including generated feedback.
    """
    qualitative_feedback = []
    for i in range(len(data)):
        if data['LLMUsage'][i] == 1:
            feedback = random.choice([
                "The LLM helped me find the bug quickly.",
                "I understood the code better with the LLM's explanation.",
                "The LLM gave me suggestions I wouldn't have thought of."
            ])
        else:
            feedback = random.choice([
                "I wish I had a tool to help me understand the code.",
                "I spent a lot of time trying to find the error myself.",
                "It was difficult to debug without assistance."
            ])
        qualitative_feedback.append(f"P{i+1}: {feedback}")

    interview_analysis = {
        "perceived_usefulness_llm": np.random.uniform(3, 5) if data['LLMUsage'].any() else np.random.uniform(1, 3),
        "anxiety_reduction_llm": np.random.uniform(1, 3) if data['LLMUsage'].any() else np.random.uniform(0, 1),
        "anxiety_reduction_herbal": np.random.uniform(1, 3) if data['HerbalBlend'].any() else np.random.uniform(0, 1),
        "qualitative_feedback": qualitative_feedback,  # Include generated feedback
    }
    return interview_analysis

# --- SEM Diagram Generation (using Matplotlib) ---

def create_sem_diagram_mpl(model_name, nodes, edges, filename):
    """Creates a conceptual SEM diagram using Matplotlib, with arrowheads.

    Args:
        model_name (str): Name of the model.
        nodes (dict): Node labels and positions: {'node_label': (x, y)}.
        edges (list of tuples): Edges: [(source, target), ...].
        filename (str): Output filename.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_facecolor('#262626')  # Ensure figure background is dark graphite
    ax.set_facecolor('#262626')   # Ensure axes background is dark graphite


    # Draw nodes (perfect circles)
    for label, pos in nodes.items():
        ax.add_patch(plt.Circle(pos, 0.3, color='#00FFFF', zorder=2))  # Cyan circles
        ax.text(pos[0], pos[1], label, color='black', ha='center', va='center', fontsize=10, zorder=3)

    # Draw edges with arrowheads
    for source, target in edges:
        x1, y1 = nodes[source]
        x2, y2 = nodes[target]
        # Use arrow instead of line for arrowheads
        ax.arrow(x1, y1, x2 - x1, y2 - y1,
                 head_width=0.15,  # Adjust arrowhead size
                 head_length=0.2,  # Adjust arrowhead length
                 fc='#00FF00',  # Arrowhead color (green)
                 ec='#00FF00',  # Edge color (green)
                 length_includes_head=True,
                 zorder=1)

    ax.set_title(f"SEM Model: {model_name}", color='#00FFFF')
    ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"SEM diagram saved to: {filename}")


# --- Statistical Plotting Functions (Matplotlib/Seaborn) ---

def create_histogram_mpl(data, column, filename):
    """Creates a histogram with the neon theme (Matplotlib)."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=False, color='#00FFFF')  # Cyan bars
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()
    print(f"Histogram saved to: {filename}")

def create_violin_plot_mpl(data, x_column, y_column, filename):
    """Creates a violin plot with the neon theme (Matplotlib/Seaborn)."""
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=data[x_column], y=data[y_column], palette="bright")
    plt.title(f"Violin Plot of {y_column} by {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.savefig(filename)
    plt.close()
    print(f"Violin plot saved to: {filename}")

def create_kde_plot_mpl(data, column1, column2, filename):
    """Creates a 2D KDE plot with the neon theme (Matplotlib/Seaborn)."""
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=data[column1], y=data[column2], cmap="coolwarm", fill=True, thresh=0, levels=100, cbar=True)
    plt.title(f"KDE Plot of {column1} vs. {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.savefig(filename)
    plt.close()
    print(f"KDE plot saved to: {filename}")

def create_stacked_bar_plot_mpl(data, x_column, y_column, color_column, filename):
    """Creates a stacked bar plot (Matplotlib/Seaborn)."""
    plt.figure(figsize=(8, 6))
    # Create a pivot table for the stacked bar plot
    pivot_data = data.groupby([x_column, color_column])[y_column].mean().unstack()
    pivot_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='cool') # Use cool colormap
    plt.title(f"Stacked Bar Plot of {y_column} by {x_column} and {color_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=color_column)
    plt.savefig(filename)
    plt.close()
    print(f"Stacked bar plot saved to: {filename}")


# --- Main Execution ---

if __name__ == '__main__':
    # Simulate data
    data = simulate_data()

    # Check if data is empty
    if data.empty:
        print("Error: Simulated data is empty.  Check the simulation function.")
        exit()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Perform statistical analyses
    statistical_results = perform_statistical_analysis(data)
    descriptive_stats, correlation_matrix, group_comparison_results = statistical_results
    print("Descriptive Statistics:\n", descriptive_stats)
    print("\nCorrelation Matrix:\n", correlation_matrix)
    print("\nGroup Comparison Results (T-tests):\n", group_comparison_results)

    # Perform regression analysis
    regression_results = perform_regression_analysis(X_train, y_train)
    print("\nRegression Results:\n", regression_results.summary())

    # Perform qualitative analyses
    prompt_analysis = analyze_prompts(data)
    interview_analysis = analyze_interviews(data)
    qualitative_results = (prompt_analysis, interview_analysis)

    # --- Apply Neon Theme Globally ---
    apply_neon_theme_mpl()

    # --- Create SEM Diagrams ---
    # Model 1: Basic Model
    nodes1 = {
        'LLM': (1, 3), 'Herbal': (1, 1), 'SelfEfficacy': (3, 3),
        'Anxiety': (3, 1), 'Performance': (5, 2)
    }
    edges1 = [('LLM', 'SelfEfficacy'), ('LLM', 'Anxiety'), ('LLM', 'Performance'),
              ('Herbal', 'Anxiety'), ('SelfEfficacy', 'Performance'), ('Anxiety', 'Performance')]
    create_sem_diagram_mpl("Model 1", nodes1, edges1, os.path.join(output_dir, "sem_model_1.png"))

    # Model 2:  With Mediators
    nodes2 = {
        'LLM': (1, 4), 'Herbal': (1, 1), 'SelfEfficacy': (3, 4),
        'Anxiety': (3, 1), 'Performance': (5, 2.5), 'EEG': (2, 5), 'EDA': (2, 0)
    }
    edges2 = [('LLM', 'SelfEfficacy'), ('LLM', 'Anxiety'), ('LLM', 'Performance'),
              ('Herbal', 'Anxiety'), ('SelfEfficacy', 'Performance'), ('Anxiety', 'Performance'),
              ('LLM', 'EEG'), ('Herbal', 'EDA'), ('EEG', 'Anxiety'), ('EDA', 'Anxiety')]
    create_sem_diagram_mpl("Model 2", nodes2, edges2, os.path.join(output_dir, "sem_model_2.png"))

    # Model 3: Full Model (Hypothetical)
    nodes3 = {
        'LLM': (1, 5), 'Herbal': (1, 1), 'SelfEfficacy': (3, 5), 'Anxiety': (3, 1),
        'Performance': (5, 3), 'EEG': (2, 6), 'EDA': (2, 0), 'POG': (6, 3),
        'InitialSE': (4, 6), 'FinalSE': (4, 4), 'InitialAnx': (4, 0), 'FinalAnx': (4, 2)
    }
    edges3 = [('LLM', 'SelfEfficacy'), ('LLM', 'Anxiety'), ('LLM', 'Performance'),
              ('Herbal', 'Anxiety'), ('SelfEfficacy', 'Performance'), ('Anxiety', 'Performance'),
              ('LLM', 'EEG'), ('Herbal', 'EDA'), ('EEG', 'Anxiety'), ('EDA', 'Anxiety'), ('POG', 'Performance'),
              ('InitialSE', 'FinalSE'), ('InitialAnx', 'FinalAnx')]
    create_sem_diagram_mpl("Model 3", nodes3, edges3, os.path.join(output_dir, "sem_model_3.png"))

    # --- Create Statistical Plots ---
    # 2 Histograms
    create_histogram_mpl(data, 'FinalSelfEfficacy', os.path.join(output_dir, 'histogram_self_efficacy.png'))
    create_histogram_mpl(data, 'FinalAnxiety', os.path.join(output_dir, 'histogram_anxiety.png'))

    # 2 Violin Plots
    create_violin_plot_mpl(data, 'LLMUsage', 'FinalSelfEfficacy', os.path.join(output_dir, 'violin_llm_self_efficacy.png'))
    create_violin_plot_mpl(data, 'HerbalBlend', 'FinalAnxiety', os.path.join(output_dir, 'violin_herbal_anxiety.png'))

    # 2 KDE Plots
    create_kde_plot_mpl(data, 'FinalSelfEfficacy', 'ErrorsIdentified', os.path.join(output_dir, 'kde_self_efficacy_errors.png'))
    create_kde_plot_mpl(data, 'FinalAnxiety', 'CompletionTime', os.path.join(output_dir, 'kde_anxiety_time.png'))

    # 2 Stacked Bar Plots
    create_stacked_bar_plot_mpl(data, 'ProgrammingExperience', 'ErrorsIdentified', 'LLMUsage', os.path.join(output_dir, 'stacked_bar_experience_errors.png'))
    create_stacked_bar_plot_mpl(data, 'Gender', 'CompletionTime', 'HerbalBlend', os.path.join(output_dir, 'stacked_bar_gender_time.png'))

    print(f"All plots saved to: {output_dir}")
