import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# ------------------------------
# Load model and tokenizer (optional)
# ------------------------------
model = None
tokenizer = None
max_len = 100
tensorflow_available = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model = load_model("feedback_model_multimodal.h5", compile=False)
    tokenizer = joblib.load("tokenizer.pkl")
    tensorflow_available = True
except ImportError:
    print("TensorFlow not available. Prediction features will be disabled.")
except Exception as e:
    print(f"Error loading model: {e}. Prediction features will be disabled.")

text_columns = [
    'teaching', 'coursecontent_text', 'examination_text',
    'labwork_text', 'library_text', 'extracurricular_text'
]

numeric_columns = [
    'teaching_score', 'coursecontent_score', 'examination_score',
    'labwork_score', 'library_score', 'extracurricular_score', 'problem_solving_score'
]

# ------------------------------
# GUI setup
# ------------------------------
root = tk.Tk()
root.title("Feedback Prediction & Data Processing - Enhanced")
root.geometry("1400x900")
root.resizable(True, True)

# Create main frames
control_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
control_frame.pack(side=tk.LEFT, fill=tk.Y)

content_frame = tk.Frame(root, padx=10, pady=10)
content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Notebook for tabs
notebook = ttk.Notebook(content_frame)
notebook.pack(fill=tk.BOTH, expand=True)

# Tab 1: Text Results
text_tab = tk.Frame(notebook)
notebook.add(text_tab, text="📄 Results")

result_text = tk.Text(text_tab, height=25, width=100, wrap="word", font=("Consolas", 10))
result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Tab 2: Visualizations
viz_tab = tk.Frame(notebook)
notebook.add(viz_tab, text="📊 Visualizations")

# Tab 3: Statistics
stats_tab = tk.Frame(notebook)
notebook.add(stats_tab, text="📈 Statistics")

df_global = None  # store uploaded dataframe
df_processed = None  # store processed dataframe

# ------------------------------
# Step 1: Upload file
# ------------------------------
def upload_file():
    global df_global
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("All supported files", "*.xlsx *.xls *.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    if not file_path:
        return
    try:
        # Determine file type and read accordingly
        if file_path.endswith('.csv'):
            df_global = pd.read_csv(file_path)
        else:
            df_global = pd.read_excel(file_path)
        
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"✅ File uploaded successfully!\n")
        result_text.insert(tk.END, f"📊 Shape: {df_global.shape[0]} rows × {df_global.shape[1]} columns\n")
        result_text.insert(tk.END, f"📁 File: {file_path}\n")
        result_text.insert(tk.END, f"📋 Columns: {list(df_global.columns)}\n")
        show_data_quality()
    except Exception as e:
        messagebox.showerror("File Error", f"Could not read file:\n{e}")

# ------------------------------
# Data Quality Check
# ------------------------------
def show_data_quality():
    global df_global
    if df_global is None:
        return
    
    result_text.insert(tk.END, f"\n🔍 Data Quality Report:\n")
    result_text.insert(tk.END, f"{'─' * 50}\n")
    
    # Missing values
    missing = df_global.isnull().sum()
    if missing.sum() > 0:
        result_text.insert(tk.END, f"⚠️  Missing Values:\n")
        for col, count in missing[missing > 0].items():
            result_text.insert(tk.END, f"   - {col}: {count} ({count/len(df_global)*100:.1f}%)\n")
    else:
        result_text.insert(tk.END, f"✅ No missing values found\n")
    
    # Data types
    result_text.insert(tk.END, f"\n📊 Data Types:\n")
    for col, dtype in df_global.dtypes.items():
        result_text.insert(tk.END, f"   - {col}: {dtype}\n")
    
    # Numeric statistics
    numeric_cols = df_global.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        result_text.insert(tk.END, f"\n📈 Numeric Statistics:\n")
        result_text.insert(tk.END, df_global[numeric_cols].describe().to_string())
    
    result_text.insert(tk.END, f"\n{'─' * 50}\n")

# ------------------------------
# Step 2: Preprocess data
# ------------------------------
def preprocess_data():
    global df_global, df_processed, numeric_columns
    if df_global is None:
        messagebox.showwarning("Warning", "Upload a file first!")
        return
    
    try:
        df = df_global.copy()
        
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"🔄 Starting preprocessing...\n")
        result_text.insert(tk.END, f"📋 Original columns: {list(df.columns)}\n")
        
        # Handle duplicate column names by adding suffixes
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        
        df.columns = [c.strip() for c in df.columns]  # remove spaces
        result_text.insert(tk.END, f"📋 Columns after deduplication: {list(df.columns)}\n")

        # Column mapping
        column_mapping = {
            'coursecontent': 'coursecontent_text',
            'coursecontent_com': 'coursecontent_score',
            'examination': 'examination_text',
            'examination_com': 'examination_score',
            'labwork': 'labwork_text',
            'labwork_com': 'labwork_score',
            'library_facilities': 'library_text',
            'library_facilities_com': 'library_score',
            'extracurricular': 'extracurricular_text',
            'problem_solving': 'problem_solving_score',
            'overall_teaching': 'overall_score',
            'teach_com': 'teaching_score',
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Handle text columns - use existing columns or create empty ones
        available_text_columns = []
        for col in text_columns:
            if col in df.columns:
                # Convert to string only if it's not already numeric
                if df[col].dtype in ['object', 'str']:
                    df[col] = df[col].astype(str).fillna('')
                    available_text_columns.append(col)
                else:
                    # If it's numeric, skip it for text processing
                    result_text.insert(tk.END, f"  ⚠️  Column '{col}' is numeric, skipping text processing\n")
            else:
                df[col] = ''
                available_text_columns.append(col)

        # Handle numeric columns - use existing columns or create with 0
        available_numeric = []
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, keep actual values
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
                if df[col].sum() > 0:  # Only add if there's actual data
                    available_numeric.append(col)
            else:
                df[col] = 0.0
        
        # If no expected columns found, use all numeric columns from the dataframe
        if not available_numeric:
            result_text.insert(tk.END, f"⚠️  Expected numeric columns not found. Using all numeric columns from data.\n")
            for col in df.columns:
                try:
                    # Try to convert to numeric
                    converted = pd.to_numeric(df[col], errors='coerce')
                    # Check if conversion was successful (not all NaN)
                    if not converted.isna().all():
                        df[col] = converted.fillna(0).astype(float)
                        if col not in available_numeric:
                            available_numeric.append(col)
                            result_text.insert(tk.END, f"  ✅ Added numeric column: {col}\n")
                    else:
                        result_text.insert(tk.END, f"  ❌ Skipped non-numeric column: {col}\n")
                except Exception as e:
                    result_text.insert(tk.END, f"  ❌ Error processing column {col}: {e}\n")
        
        # Update global numeric_columns to only include available columns
        if available_numeric:
            numeric_columns = available_numeric
            result_text.insert(tk.END, f"✅ Found {len(numeric_columns)} numeric columns with data\n")
            result_text.insert(tk.END, f"📋 Numeric columns: {numeric_columns}\n")
        else:
            result_text.insert(tk.END, f"❌ No numeric columns with data found!\n")
        
        # Combine text columns (only if there are actual text columns)
        if available_text_columns:
            df['all_text'] = df[available_text_columns].agg(' '.join, axis=1)
            result_text.insert(tk.END, f"✅ Combined {len(available_text_columns)} text columns\n")
        else:
            # If no text columns, create empty text column
            df['all_text'] = ''
            result_text.insert(tk.END, f"⚠️  No text columns found, created empty 'all_text' column\n")

        df_processed = df
        result_text.insert(tk.END, f"\n✅ df_processed set successfully with {len(df_processed)} rows\n")
        result_text.insert(tk.END, f"✅ Data preprocessing completed!\n")
        result_text.insert(tk.END, f"📊 Shape: {df_processed.shape}\n")
        result_text.insert(tk.END, f"🔤 Text features combined into 'all_text'\n")
        result_text.insert(tk.END, f"🔢 Numeric features available: {len(numeric_columns)}\n")
        result_text.insert(tk.END, f"📋 Available numeric columns: {numeric_columns}\n")
        result_text.insert(tk.END, f"📋 All columns: {list(df_processed.columns)}\n")
        
        # Show sample of numeric data
        if len(numeric_columns) > 0:
            result_text.insert(tk.END, f"\n📊 Sample numeric data:\n")
            result_text.insert(tk.END, df_processed[numeric_columns].head().to_string())
        
        result_text.insert(tk.END, f"\n{'=' * 60}\n")
        result_text.insert(tk.END, f"✅ Ready for analysis! You can now use:\n")
        result_text.insert(tk.END, f"   - Preview Data\n")
        result_text.insert(tk.END, f"   - Show Statistics\n")
        result_text.insert(tk.END, f"   - Show Visualizations\n")
        result_text.insert(tk.END, f"   - Feature Importance\n")
        result_text.insert(tk.END, f"{'=' * 60}\n")
        
        show_statistics()
    except Exception as e:
        messagebox.showerror("Preprocessing Error", f"Error during preprocessing:\n{e}")
        result_text.insert(tk.END, f"\n❌ Preprocessing failed: {e}\n")

# ------------------------------
# Show Statistics
# ------------------------------
def show_statistics():
    global df_processed
    if df_processed is None:
        return
    
    try:
        # Clear statistics tab
        for widget in stats_tab.winfo_children():
            widget.destroy()
        
        # Create statistics text widget
        stats_text = tk.Text(stats_tab, height=30, width=100, wrap="word", font=("Consolas", 10))
        stats_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        stats_text.insert(tk.END, f"📊 Statistical Summary\n")
        stats_text.insert(tk.END, f"{'=' * 60}\n\n")
        
        # Overall statistics
        stats_text.insert(tk.END, f"Total Records: {len(df_processed)}\n")
        stats_text.insert(tk.END, f"Total Features: {len(df_processed.columns)}\n\n")
        
        # Get actual numeric columns from the dataframe
        actual_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_text.insert(tk.END, f"Detected numeric columns: {actual_numeric_cols}\n\n")
        
        if len(actual_numeric_cols) > 0:
            # Numeric features statistics
            numeric_cols = df_processed[actual_numeric_cols]
            stats_text.insert(tk.END, f"🔢 Numeric Features Statistics:\n")
            stats_text.insert(tk.END, f"{'─' * 60}\n")
            stats_text.insert(tk.END, numeric_cols.describe().to_string())
            stats_text.insert(tk.END, f"\n\n")
            
            # Correlation matrix
            if len(actual_numeric_cols) > 1:
                stats_text.insert(tk.END, f"🔗 Correlation Matrix:\n")
                stats_text.insert(tk.END, f"{'─' * 60}\n")
                try:
                    correlation = numeric_cols.corr()
                    # Replace NaN with 0 for cleaner display
                    correlation = correlation.fillna(0)
                    stats_text.insert(tk.END, correlation.to_string())
                except Exception as e:
                    stats_text.insert(tk.END, f"Error calculating correlation: {e}")
                stats_text.insert(tk.END, f"\n\n")
        else:
            stats_text.insert(tk.END, f"⚠️  No numeric columns found in the data\n\n")
        
        # Text features statistics
        stats_text.insert(tk.END, f"🔤 Text Features Statistics:\n")
        stats_text.insert(tk.END, f"{'─' * 60}\n")
        for col in text_columns:
            if col in df_processed.columns:
                avg_length = df_processed[col].astype(str).str.len().mean()
                non_empty = df_processed[col].astype(str).str.strip().astype(bool).sum()
                stats_text.insert(tk.END, f"{col}: Avg length = {avg_length:.1f} chars, Non-empty = {non_empty}\n")
        
        stats_text.insert(tk.END, f"\n{'=' * 60}\n")
    except Exception as e:
        messagebox.showerror("Statistics Error", f"Error generating statistics:\n{e}")

# ------------------------------
# Step 3: Preview data
# ------------------------------
def preview_data():
    global df_processed
    if df_processed is None:
        messagebox.showwarning("Warning", "Preprocess the data first!")
        return
    try:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"📄 Preview of processed data:\n")
        result_text.insert(tk.END, f"{'─' * 60}\n")
        result_text.insert(tk.END, df_processed.head(10).to_string())
        result_text.insert(tk.END, f"\n{'─' * 60}\n")
        result_text.insert(tk.END, f"✅ Data preview successful!\n")
    except Exception as e:
        messagebox.showerror("Preview Error", f"Error previewing data:\n{e}")

# ------------------------------
# Step 4: Predict
# ------------------------------
def predict():
    global df_processed
    if df_processed is None:
        messagebox.showwarning("Warning", "Preprocess the data first!")
        return
    
    try:
        if tensorflow_available:
            # Use TensorFlow model if available
            X_numeric = df_processed[numeric_columns].values
            try:
                scaler = joblib.load("scaler.pkl")
                X_numeric = scaler.transform(X_numeric)
            except:
                pass

            sequences = tokenizer.texts_to_sequences(df_processed['all_text'])
            X_text = pad_sequences(sequences, maxlen=max_len, padding='post')

            preds = model.predict([X_text, X_numeric])
            df_processed['predicted_overall_score'] = preds.flatten()
            
            prediction_method = "Deep Learning Model"
        else:
            # Use simple weighted average algorithm when TensorFlow is not available
            if len(numeric_columns) > 0:
                try:
                    # Calculate weighted average of numeric columns
                    # Equal weights for all columns
                    weights = [1.0 / len(numeric_columns)] * len(numeric_columns)
                    df_processed['predicted_overall_score'] = (
                        df_processed[numeric_columns].values * weights
                    ).sum(axis=1)
                    
                    # Scale to 0-5 range if needed
                    min_pred = df_processed['predicted_overall_score'].min()
                    max_pred = df_processed['predicted_overall_score'].max()
                    if max_pred > min_pred:
                        df_processed['predicted_overall_score'] = (
                            (df_processed['predicted_overall_score'] - min_pred) / (max_pred - min_pred) * 5
                    )
                    else:
                        # If all values are the same, set to middle value
                        df_processed['predicted_overall_score'] = 2.5
                    
                    prediction_method = "Simple Weighted Average"
                except Exception as e:
                    messagebox.showerror("Prediction Error", f"Error in simple prediction algorithm:\n{e}")
                    return
            else:
                messagebox.showwarning("Warning", "No numeric columns available for prediction!")
                return

        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"📊 Prediction Results ({prediction_method}):\n\n")
        result_text.insert(tk.END, f"{'─' * 60}\n")
        result_text.insert(tk.END, f"Total Predictions: {len(df_processed)}\n")
        result_text.insert(tk.END, f"Average Predicted Score: {df_processed['predicted_overall_score'].mean():.2f}\n")
        result_text.insert(tk.END, f"Min Predicted Score: {df_processed['predicted_overall_score'].min():.2f}\n")
        result_text.insert(tk.END, f"Max Predicted Score: {df_processed['predicted_overall_score'].max():.2f}\n")
        result_text.insert(tk.END, f"Std Dev: {df_processed['predicted_overall_score'].std():.2f}\n")
        result_text.insert(tk.END, f"{'─' * 60}\n\n")
        
        # Show only first 10 and last 5 predictions to avoid too much output
        if len(df_processed) <= 15:
            for idx, row in df_processed.iterrows():
                result_text.insert(tk.END, f"Row {idx+1} -> Predicted Overall Score: {row['predicted_overall_score']:.2f}\n")
        else:
            result_text.insert(tk.END, f"First 10 predictions:\n")
            for idx in range(10):
                result_text.insert(tk.END, f"Row {idx+1} -> Predicted Overall Score: {df_processed.iloc[idx]['predicted_overall_score']:.2f}\n")
            result_text.insert(tk.END, f"\n... ({len(df_processed) - 15} more predictions) ...\n\n")
            result_text.insert(tk.END, f"Last 5 predictions:\n")
            for idx in range(len(df_processed) - 5, len(df_processed)):
                result_text.insert(tk.END, f"Row {idx+1} -> Predicted Overall Score: {df_processed.iloc[idx]['predicted_overall_score']:.2f}\n")
        
        result_text.insert(tk.END, f"\n💡 Use 'Export Results' to save all predictions to file\n")
        
        # Generate visualizations
        show_visualizations()
        show_statistics()
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction:\n{e}")

# ------------------------------
# Show Visualizations
# ------------------------------
def show_visualizations():
    global df_processed
    if df_processed is None:
        messagebox.showwarning("Warning", "Process data first!")
        return
    
    try:
        # Clear visualization tab
        for widget in viz_tab.winfo_children():
            widget.destroy()
        
        # Get actual numeric columns
        actual_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(actual_numeric_cols) == 0:
            # Show message if no numeric data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No numeric data available for visualization', 
                    ha='center', va='center', fontsize=14, color='red')
            ax.set_title('Data Visualization', fontweight='bold')
            canvas = FigureCanvasTkAgg(fig, master=viz_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feedback Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. First Numeric Feature Distribution
        if len(actual_numeric_cols) > 0:
            first_col = actual_numeric_cols[0]
            data = df_processed[first_col].dropna()
            if len(data) > 0:
                axes[0, 0].hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                axes[0, 0].set_title(f'Distribution of {first_col.replace("_", " ").title()}', fontweight='bold')
                axes[0, 0].set_xlabel(first_col.replace('_', ' ').title())
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        # 2. Numeric Features Box Plot
        numeric_data = df_processed[actual_numeric_cols]
        if len(actual_numeric_cols) <= 10:  # Only show boxplot if not too many columns
            try:
                box_data = [numeric_data[col].dropna() for col in actual_numeric_cols]
                # Filter out empty datasets
                box_data = [d for d in box_data if len(d) > 0]
                box_labels = [col.replace('_', ' ').title()[:15] for col, d in zip(actual_numeric_cols, [numeric_data[col].dropna() for col in actual_numeric_cols]) if len(d) > 0]
                
                if len(box_data) > 0:
                    axes[0, 1].boxplot(box_data, labels=box_labels)
                    axes[0, 1].set_title('Numeric Features Distribution', fontweight='bold')
                    axes[0, 1].set_ylabel('Value')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'Boxplot error: {str(e)[:25]}', ha='center', va='center', fontsize=10)
                axes[0, 1].set_title('Numeric Features Distribution', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, f'Too many columns ({len(actual_numeric_cols)}) for boxplot', 
                           ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Numeric Features Distribution', fontweight='bold')
        
        # 3. Correlation Heatmap
        if len(actual_numeric_cols) > 1:
            try:
                correlation = numeric_data.corr()
                # Replace NaN with 0 to prevent warnings
                correlation = correlation.fillna(0)
                if not correlation.isna().all().all():
                    im = axes[1, 0].imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    axes[1, 0].set_xticks(range(len(actual_numeric_cols)))
                    axes[1, 0].set_yticks(range(len(actual_numeric_cols)))
                    axes[1, 0].set_xticklabels([col.replace('_', ' ').title()[:10] for col in actual_numeric_cols], rotation=45, ha='right')
                    axes[1, 0].set_yticklabels([col.replace('_', ' ').title()[:10] for col in actual_numeric_cols])
                    axes[1, 0].set_title('Feature Correlation Heatmap', fontweight='bold')
                    
                    # Add correlation values
                    for i in range(len(actual_numeric_cols)):
                        for j in range(len(actual_numeric_cols)):
                            val = correlation.iloc[i, j]
                            if not pd.isna(val):
                                text = axes[1, 0].text(j, i, f'{val:.2f}',
                                                      ha="center", va="center", color="black", fontsize=8)
                    
                    plt.colorbar(im, ax=axes[1, 0])
                else:
                    axes[1, 0].text(0.5, 0.5, 'No valid correlation data', ha='center', va='center')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Correlation error: {str(e)[:30]}', ha='center', va='center', fontsize=10)
                axes[1, 0].set_title('Feature Correlation Heatmap', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'Need at least 2 numeric columns for correlation', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Feature Correlation Heatmap', fontweight='bold')
        
        # 4. Scatter plot of first two numeric features
        if len(actual_numeric_cols) >= 2:
            x_col = actual_numeric_cols[0]
            y_col = actual_numeric_cols[1]
            x_data = df_processed[x_col].dropna()
            y_data = df_processed[y_col].dropna()
            if len(x_data) > 0 and len(y_data) > 0:
                axes[1, 1].scatter(x_data, y_data, alpha=0.6, color='coral')
                axes[1, 1].set_title(f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}', fontweight='bold')
                axes[1, 1].set_xlabel(x_col.replace('_', ' ').title())
                axes[1, 1].set_ylabel(y_col.replace('_', ' ').title())
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No data available', ha='center', va='center')
        elif 'predicted_overall_score' in df_processed.columns and len(actual_numeric_cols) >= 1:
            # If predictions exist, show prediction vs first feature
            feature = actual_numeric_cols[0]
            x_data = df_processed[feature].dropna()
            y_data = df_processed['predicted_overall_score'].dropna()
            if len(x_data) > 0 and len(y_data) > 0:
                axes[1, 1].scatter(x_data, y_data, alpha=0.6, color='coral')
                axes[1, 1].set_title(f'Predicted Score vs {feature.replace("_", " ").title()}', fontweight='bold')
                axes[1, 1].set_xlabel(feature.replace('_', ' ').title())
                axes[1, 1].set_ylabel('Predicted Score')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No data available', ha='center', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'Need more data for scatter plot', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Feature Scatter Plot', fontweight='bold')
        
        plt.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=viz_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    except Exception as e:
        messagebox.showerror("Visualization Error", f"Error generating visualizations:\n{e}")
        # Show error message in tab
        for widget in viz_tab.winfo_children():
            widget.destroy()
        error_label = tk.Label(viz_tab, text=f"Error: {str(e)}", fg="red", font=("Arial", 12))
        error_label.pack(pady=20)

# ------------------------------
# Export Results
# ------------------------------
def export_results():
    global df_processed
    if df_processed is None or 'predicted_overall_score' not in df_processed.columns:
        messagebox.showwarning("Warning", "No predictions to export. Run predictions first!")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not file_path:
        return
    
    try:
        if file_path.endswith('.csv'):
            df_processed.to_csv(file_path, index=False)
        else:
            df_processed.to_excel(file_path, index=False)
        
        messagebox.showinfo("Export Success", f"Results exported successfully to:\n{file_path}")
        result_text.insert(tk.END, f"\n✅ Results exported to: {file_path}\n")
    except Exception as e:
        messagebox.showerror("Export Error", f"Could not export results:\n{e}")

# ------------------------------
# Export Statistics Report
# ------------------------------
def export_statistics():
    global df_processed
    if df_processed is None:
        messagebox.showwarning("Warning", "No data to export. Process data first!")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if not file_path:
        return
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FEEDBACK PREDICTION STATISTICS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Records: {len(df_processed)}\n")
            f.write(f"Total Features: {len(df_processed.columns)}\n\n")
            
            if 'predicted_overall_score' in df_processed.columns:
                f.write("PREDICTION STATISTICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Average Predicted Score: {df_processed['predicted_overall_score'].mean():.4f}\n")
                f.write(f"Min Predicted Score: {df_processed['predicted_overall_score'].min():.4f}\n")
                f.write(f"Max Predicted Score: {df_processed['predicted_overall_score'].max():.4f}\n")
                f.write(f"Std Dev: {df_processed['predicted_overall_score'].std():.4f}\n\n")
            
            f.write("NUMERIC FEATURES STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(df_processed[numeric_columns].describe().to_string())
            f.write("\n\n")
            
            f.write("CORRELATION MATRIX\n")
            f.write("-" * 60 + "\n")
            f.write(df_processed[numeric_columns].corr().to_string())
            f.write("\n\n")
            
            f.write("=" * 60 + "\n")
        
        messagebox.showinfo("Export Success", f"Statistics report exported to:\n{file_path}")
        result_text.insert(tk.END, f"\n✅ Statistics report exported to: {file_path}\n")
    except Exception as e:
        messagebox.showerror("Export Error", f"Could not export statistics:\n{e}")

# ------------------------------
# Feature Importance Analysis
# ------------------------------
def analyze_feature_importance():
    global df_processed
    if df_processed is None:
        messagebox.showwarning("Warning", "Process data first!")
        return
    
    # Get actual numeric columns
    actual_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(actual_numeric_cols) == 0:
        messagebox.showwarning("Warning", "No numeric columns available for analysis!")
        return
    
    # Calculate correlation with predicted score if available, otherwise use variance
    correlations = {}
    if 'predicted_overall_score' in df_processed.columns:
        for col in actual_numeric_cols:
            if col != 'predicted_overall_score':
                corr = df_processed[col].corr(df_processed['predicted_overall_score'])
                correlations[col] = abs(corr) if not pd.isna(corr) else 0.0
    else:
        # Use standard deviation as importance measure if no predictions
        for col in actual_numeric_cols:
            std = df_processed[col].std()
            correlations[col] = std if not pd.isna(std) else 0.0
    
    # Sort by importance
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Display in results tab (clear previous content first)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"🔍 Feature Importance Analysis:\n")
    result_text.insert(tk.END, f"{'─' * 60}\n")
    if 'predicted_overall_score' in df_processed.columns:
        result_text.insert(tk.END, f"Based on correlation with predicted score\n")
    else:
        result_text.insert(tk.END, f"Based on standard deviation (no predictions yet)\n")
    result_text.insert(tk.END, f"{'─' * 60}\n")
    for feature, importance in sorted_features:
        result_text.insert(tk.END, f"{feature.replace('_', ' ').title():30s}: {importance:.4f}\n")
    result_text.insert(tk.END, f"{'─' * 60}\n")
    
    # Create visualization
    for widget in viz_tab.winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    features = [f.replace('_', ' ').title() for f, _ in sorted_features]
    importances = [imp for _, imp in sorted_features]
    
    bars = ax.barh(features, importances, color='steelblue')
    if 'predicted_overall_score' in df_processed.columns:
        ax.set_xlabel('Absolute Correlation with Predicted Score', fontweight='bold')
    else:
        ax.set_xlabel('Standard Deviation', fontweight='bold')
    ax.set_title('Feature Importance Analysis', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=viz_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ------------------------------
# Buttons
# ------------------------------
# Main workflow buttons
tk.Label(control_frame, text="MAIN WORKFLOW", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(0, 5))
tk.Button(control_frame, text="1️⃣ Upload File", command=upload_file, bg="#4CAF50", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)
tk.Button(control_frame, text="2️⃣ Preprocess Data", command=preprocess_data, bg="#4CAF50", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)
tk.Button(control_frame, text="3️⃣ Preview Data", command=preview_data, bg="#4CAF50", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)

if tensorflow_available:
    tk.Button(control_frame, text="4️⃣ Predict (ML Model)", command=predict, bg="#4CAF50", fg="white", 
              font=("Arial", 10), width=20).pack(pady=3)
else:
    tk.Button(control_frame, text="4️⃣ Predict (Simple)", command=predict, bg="#4CAF50", fg="white", 
              font=("Arial", 10), width=20).pack(pady=3)

# Analysis buttons
tk.Label(control_frame, text="ANALYSIS TOOLS", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(15, 5))
tk.Button(control_frame, text="📊 Feature Importance", command=analyze_feature_importance, bg="#2196F3", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)
tk.Button(control_frame, text="📈 Show Statistics", command=show_statistics, bg="#2196F3", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)
tk.Button(control_frame, text="🎨 Show Visualizations", command=show_visualizations, bg="#2196F3", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)

# Export buttons
tk.Label(control_frame, text="EXPORT OPTIONS", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(15, 5))
tk.Button(control_frame, text="💾 Export Results", command=export_results, bg="#FF9800", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)
tk.Button(control_frame, text="📄 Export Statistics", command=export_statistics, bg="#FF9800", fg="white", 
          font=("Arial", 10), width=20).pack(pady=3)

# Info label
tk.Label(control_frame, text="Feedback Prediction\nSystem v2.0", bg="#f0f0f0", 
          font=("Arial", 8), fg="#666").pack(side=tk.BOTTOM, pady=10)

root.mainloop()
