import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class GenericChurnPredictor:
    """
    Generic Customer Churn Predictor for ANY Business
    Works with any dataset - just needs a 'churn' column (or similar)
    """
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        self.feature_columns = []
        self.target_column = None
        self.training_data = None
        self.categorical_columns = []
        self.numerical_columns = []
        
    def detect_target_column(self, df):
        """Automatically detect the churn/target column"""
        possible_targets = ['churn', 'churned', 'is_churn', 'customer_churn', 'attrition', 
                           'left', 'cancelled', 'canceled', 'retention', 'stayed', 'active']
        
        # Check for exact matches (case insensitive)
        for col in df.columns:
            if col.lower() in possible_targets:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for target in possible_targets:
                if target in col.lower():
                    return col
        
        # Check for binary columns (0/1 or Yes/No)
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                if len(unique_vals) == 2:
                    if set(map(str, unique_vals)).issubset({'0', '1', 'Yes', 'No', 'True', 'False', 'Y', 'N'}):
                        return col
            elif df[col].dtype in ['int64', 'float64']:
                unique_vals = df[col].unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    return col
        
        return None
    
    def prepare_data(self, df, target_column=None):
        """Prepare any dataset for churn prediction"""
        if target_column is None:
            target_column = self.detect_target_column(df)
            
        if target_column is None:
            raise ValueError("Could not detect target column. Please specify the column name that indicates churn.")
        
        self.target_column = target_column
        
        # Make a copy
        df_processed = df.copy()
        
        # Convert target to binary (0/1)
        if df_processed[target_column].dtype == 'object':
            # Handle text values
            positive_values = ['yes', 'y', 'true', '1', 'churn', 'churned', 'left', 'cancelled', 'canceled']
            df_processed[target_column] = df_processed[target_column].astype(str).str.lower().isin(positive_values).astype(int)
        else:
            # Handle numeric values
            df_processed[target_column] = (df_processed[target_column] > 0).astype(int)
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Remove ID columns (typically first column or columns with 'id' in name)
        id_columns = []
        for col in X.columns:
            if ('id' in col.lower() or 
                col.lower().startswith('customer') and 'id' in col.lower() or
                X[col].dtype == 'object' and X[col].nunique() == len(X)):
                id_columns.append(col)
        
        if id_columns:
            X = X.drop(columns=id_columns)
            st.info(f"Removed ID columns: {id_columns}")
        
        # Identify column types
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        return X, y
    
    def preprocess_features(self, X, is_training=True):
        """Preprocess features for any dataset"""
        X_processed = X.copy()
        
        # Handle missing values
        if len(self.numerical_columns) > 0:
            if is_training:
                X_processed[self.numerical_columns] = self.imputer_numeric.fit_transform(X_processed[self.numerical_columns])
            else:
                X_processed[self.numerical_columns] = self.imputer_numeric.transform(X_processed[self.numerical_columns])
        
        if len(self.categorical_columns) > 0:
            if is_training:
                X_processed[self.categorical_columns] = self.imputer_categorical.fit_transform(X_processed[self.categorical_columns])
            else:
                X_processed[self.categorical_columns] = self.imputer_categorical.transform(X_processed[self.categorical_columns])
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in X_processed.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        try:
                            X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
                        except ValueError:
                            # Replace unseen categories with most frequent
                            most_frequent = self.label_encoders[col].classes_[0]
                            X_processed[col] = X_processed[col].astype(str).apply(
                                lambda x: x if x in self.label_encoders[col].classes_ else most_frequent
                            )
                            X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        return X_processed
    
    def train_models(self, df, target_column=None):
        """Train ML models on any churn dataset"""
        try:
            # Prepare data
            X, y = self.prepare_data(df, target_column)
            
            # Store training data
            self.training_data = df.copy()
            
            # Preprocess features
            X_processed = self.preprocess_features(X, is_training=True)
            
            # Store feature columns
            self.feature_columns = X_processed.columns.tolist()
            
            # Check class distribution
            class_counts = y.value_counts()
            churn_rate = class_counts.get(1, 0) / len(y)
            
            st.write(f"📊 **Dataset Analysis:**")
            st.write(f"- Total customers: {len(y):,}")
            st.write(f"- Churned customers: {class_counts.get(1, 0):,}")
            st.write(f"- Retained customers: {class_counts.get(0, 0):,}")
            st.write(f"- Churn rate: {churn_rate:.1%}")
            st.write(f"- Features: {len(self.feature_columns)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=200, 
                    random_state=42, 
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced'
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    random_state=42, 
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=8,
                    min_samples_split=5
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    class_weight='balanced'
                )
            }
            
            results = {}
            
            for name, model in models.items():
                try:
                    st.write(f"🔄 Training {name}...")
                    
                    # Train model
                    if name == 'Logistic Regression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'auc_score': auc_score,
                        'cv_score': cv_mean,
                        'cv_std': cv_std,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                    
                    self.models[name] = model
                    
                    st.write(f"✅ {name} - AUC: {auc_score:.3f}, Accuracy: {accuracy:.3f}, CV: {cv_mean:.3f} ± {cv_std:.3f}")
                    
                except Exception as e:
                    st.warning(f"❌ Error training {name}: {str(e)}")
                    continue
            
            if results:
                # Select best model based on AUC score
                best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
                self.best_model = results[best_model_name]['model']
                self.best_model_name = best_model_name
                self.is_trained = True
                
                st.success(f"🏆 **Best Model:** {best_model_name} (AUC: {results[best_model_name]['auc_score']:.3f})")
            
            return results
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return {}
    
    def predict_churn(self, customer_data):
        """Predict churn for new customer data"""
        if not self.is_trained:
            st.error("❌ Model not trained yet!")
            return None
            
        try:
            # Convert to DataFrame if dict
            if isinstance(customer_data, dict):
                customer_df = pd.DataFrame([customer_data])
            else:
                customer_df = customer_data.copy()
            
            # Add missing columns with default values
            for col in self.feature_columns:
                if col not in customer_df.columns:
                    if col in self.numerical_columns:
                        customer_df[col] = 0
                    else:
                        customer_df[col] = 'Unknown'
            
            # Select and order columns
            customer_df = customer_df[self.feature_columns]
            
            # Preprocess
            customer_processed = self.preprocess_features(customer_df, is_training=False)
            
            # Make prediction
            if self.best_model_name == 'Logistic Regression':
                customer_scaled = self.scaler.transform(customer_processed)
                churn_probability = self.best_model.predict_proba(customer_scaled)[0, 1]
            else:
                churn_probability = self.best_model.predict_proba(customer_processed)[0, 1]
            
            return churn_probability
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.is_trained or not hasattr(self.best_model, 'feature_importances_'):
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    st.set_page_config(page_title="Generic Churn Predictor", page_icon="📊", layout="wide")
    
    st.markdown("""
    # 📊 Universal Customer Churn Predictor
    
    **Upload ANY business dataset and predict customer churn!**
    
    This system works with data from:
    - 🏦 Banks & Financial Services
    - 📱 Telecom & SaaS Companies  
    - 🛒 E-commerce & Retail
    - 🏥 Healthcare & Insurance
    - 🎓 Education & Subscriptions
    - 🏨 Hotels & Services
    
    Just upload your CSV with customer data and a churn indicator column!
    """)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = GenericChurnPredictor()
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox("Choose Option", 
                               ["📤 Upload & Train", "🔮 Predict Churn", "📊 Model Analysis", "💡 How to Use"])
    
    if page == "📤 Upload & Train":
        st.header("📤 Upload Your Dataset & Train ML Models")
        
        st.info("""
        **📋 Data Requirements:**
        - CSV file with customer data
        - One column indicating churn (Yes/No, 1/0, True/False, etc.)
        - Customer features (age, tenure, spending, etc.)
        
        **🔍 The system will automatically:**
        - Detect your churn column
        - Handle missing values
        - Encode categorical variables
        - Train multiple ML models
        - Select the best performer
        """)
        
        uploaded_file = st.file_uploader("Choose your CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ **File loaded successfully!** ({len(df)} rows, {len(df.columns)} columns)")
                
                # Show data preview
                st.subheader("🔍 Data Preview")
                st.dataframe(df.head())
                
                # Show basic info
                st.subheader("📊 Dataset Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                
                # Let user select target column
                st.subheader("🎯 Select Target Column")
                
                # Try to auto-detect
                auto_detected = st.session_state.predictor.detect_target_column(df)
                if auto_detected:
                    st.success(f"✅ **Auto-detected churn column:** `{auto_detected}`")
                    default_idx = list(df.columns).index(auto_detected)
                else:
                    st.warning("⚠️ Could not auto-detect churn column. Please select manually.")
                    default_idx = 0
                
                target_column = st.selectbox(
                    "Select the column that indicates customer churn:",
                    options=df.columns.tolist(),
                    index=default_idx
                )
                
                # Show target distribution
                if target_column:
                    st.write(f"**Target column distribution:**")
                    target_counts = df[target_column].value_counts()
                    st.write(target_counts)
                
                # Train models
                if st.button("🚀 Train Machine Learning Models"):
                    with st.spinner("🔄 Training models on your data..."):
                        results = st.session_state.predictor.train_models(df, target_column)
                        st.session_state.training_results = results
                        st.session_state.uploaded_data = df
                    
                    if results:
                        st.success("🎉 **Models trained successfully!**")
                        
                        # Show results
                        st.subheader("🏆 Model Performance")
                        
                        perf_data = []
                        for name, result in results.items():
                            perf_data.append({
                                'Model': name,
                                'AUC Score': f"{result['auc_score']:.3f}",
                                'Accuracy': f"{result['accuracy']:.3f}",
                                'Precision': f"{result['precision']:.3f}",
                                'Recall': f"{result['recall']:.3f}",
                                'CV Score': f"{result['cv_score']:.3f} ± {result['cv_std']:.3f}"
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df)
                        
                        # Feature importance
                        importance_df = st.session_state.predictor.get_feature_importance()
                        if importance_df is not None:
                            st.subheader("🎯 Most Important Features")
                            
                            fig = px.bar(importance_df.head(10), 
                                       x='importance', y='feature',
                                       orientation='h', 
                                       title="Top 10 Features for Churn Prediction")
                            st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.write("Please ensure your CSV file is properly formatted.")
    
    elif page == "🔮 Predict Churn":
        st.header("🔮 Predict Customer Churn")
        
        if not st.session_state.predictor.is_trained:
            st.warning("⚠️ **Please upload data and train models first!**")
            st.stop()
        
        st.info("**Enter customer information to predict churn probability using your trained ML model.**")
        
        # Dynamic form based on training data
        if hasattr(st.session_state, 'uploaded_data'):
            df = st.session_state.uploaded_data
            
            st.subheader("📝 Customer Information")
            
            # Create input form
            customer_data = {}
            
            # Get feature columns (excluding target)
            feature_cols = [col for col in df.columns if col != st.session_state.predictor.target_column]
            
            # Create input fields
            for col in feature_cols:
                if col in st.session_state.predictor.numerical_columns:
                    # Numerical input
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    
                    customer_data[col] = st.number_input(
                        f"{col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{col}"
                    )
                else:
                    # Categorical input
                    unique_vals = df[col].unique().tolist()
                    customer_data[col] = st.selectbox(
                        f"{col}",
                        options=unique_vals,
                        key=f"input_{col}"
                    )
            
            if st.button("🔮 Predict Churn Probability"):
                churn_prob = st.session_state.predictor.predict_churn(customer_data)
                
                if churn_prob is not None:
                    st.success("🎯 **Prediction Complete!**")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Churn Probability", f"{churn_prob:.1%}")
                    
                    with col2:
                        risk_level = "🔴 HIGH" if churn_prob > 0.7 else "🟡 MEDIUM" if churn_prob > 0.4 else "🟢 LOW"
                        st.metric("Risk Level", risk_level)
                    
                    with col3:
                        confidence = "High" if abs(churn_prob - 0.5) > 0.3 else "Medium" if abs(churn_prob - 0.5) > 0.2 else "Low"
                        st.metric("Confidence", confidence)
                    
                    # Recommendations
                    st.subheader("💡 Recommended Actions")
                    
                    if churn_prob > 0.7:
                        st.error("🚨 **HIGH RISK - Immediate Action Required**")
                        recommendations = [
                            "📞 Schedule immediate customer retention call",
                            "💰 Offer personalized retention package",
                            "🎁 Provide premium service upgrade",
                            "📧 Send executive-level attention",
                            "⭐ Assign dedicated account manager"
                        ]
                    elif churn_prob > 0.4:
                        st.warning("⚡ **MEDIUM RISK - Proactive Engagement**")
                        recommendations = [
                            "📋 Send satisfaction survey",
                            "🎁 Offer loyalty program enrollment",
                            "📊 Monitor usage patterns",
                            "💬 Proactive customer service",
                            "📱 Suggest service optimizations"
                        ]
                    else:
                        st.success("✅ **LOW RISK - Maintain Current Strategy**")
                        recommendations = [
                            "🌟 Consider for upselling",
                            "📝 Use as reference customer",
                            "🎯 Include in referral program",
                            "📈 Monitor for expansion",
                            "💎 Maintain service quality"
                        ]
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
    
    elif page == "📊 Model Analysis":
        st.header("📊 Model Performance Analysis")
        
        if not st.session_state.predictor.is_trained:
            st.warning("⚠️ **Please train models first!**")
            st.stop()
        
        if hasattr(st.session_state, 'training_results'):
            results = st.session_state.training_results
            
            # Model comparison
            st.subheader("🔍 Detailed Model Comparison")
            
            # Performance metrics
            metrics_data = []
            for name, result in results.items():
                metrics_data.append({
                    'Model': name,
                    'AUC Score': result['auc_score'],
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'CV Score': result['cv_score']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Visualizations
            fig = px.bar(metrics_df, x='Model', y='AUC Score', 
                        title="Model Performance Comparison (AUC Score)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            importance_df = st.session_state.predictor.get_feature_importance()
            if importance_df is not None:
                st.subheader("🎯 Feature Importance Analysis")
                
                fig = px.bar(importance_df.head(15), 
                           x='importance', y='feature',
                           orientation='h', 
                           title="Top 15 Most Important Features")
                st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            st.subheader("📈 ROC Curve Analysis")
            
            best_result = results[st.session_state.predictor.best_model_name]
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(best_result['y_test'], best_result['y_pred_proba'])
            
            fig = px.line(x=fpr, y=tpr, 
                         title=f"ROC Curve - {st.session_state.predictor.best_model_name}")
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, 
                         line=dict(dash='dash', color='gray'))
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "💡 How to Use":
        st.header("💡 How to Use This Churn Predictor")
        
        st.markdown("""
        ## 🎯 What This Tool Does
        
        This is a **universal customer churn predictor** that works with ANY business dataset. It uses machine learning to:
        - Analyze customer behavior patterns
        - Predict which customers are likely to churn
        - Provide actionable retention recommendations
        
        ## 📋 Data Requirements
        
        Your CSV file should contain:
        
        ### ✅ Required:
        - **Customer records** (one row per customer)
        - **Churn indicator** column (Yes/No, 1/0, True/False, etc.)
        - **Customer features** (demographics, behavior, usage, etc.)
        
        ### 📊 Example Data Structure:
        ```
        CustomerID | Age | Tenure | MonthlySpend | ProductsUsed | Churn
        CUST001    | 34  | 12     | 150.50       | 3            | No
        CUST002    | 45  | 6      | 89.20        | 1            | Yes
        CUST003    | 28  | 24     | 220.75       | 5            | No
        ```
        
        ## 🚀 Step-by-Step Process
        
        ### 1. **Upload Your Data**
        - Go to "📤 Upload & Train" tab
        - Upload your CSV file
        - System automatically detects churn column
        
        ### 2. **Train Models**
        - Click "🚀 Train Machine Learning Models"
        - System trains 3 different ML algorithms
        - Selects best performing model
        
        ### 3. **Make Predictions**
        - Go to "🔮 Predict Churn" tab
        - Enter customer information
        - Get instant churn probability
        
        ### 4. **Analyze Results**
        - View model performance metrics
        - Understand feature importance
        - Get actionable recommendations
        
        ## 🏢 Business Use Cases
        
        ### 🏦 **Banking & Finance**
        - Predict account closures
        - Identify at-risk credit card customers
        - Optimize retention campaigns
        
        ### 📱 **SaaS & Telecom**
        - Predict subscription cancellations
        - Identify service downgrades
        - Optimize pricing strategies
        
        ### 🛒 **E-commerce & Retail**
        - Predict customer lifetime value
        - Identify inactive customers
        - Optimize loyalty programs
        
        ### 🏥 **Healthcare & Insurance**
        - Predict policy cancellations
        - Identify patient no-shows
        - Optimize care programs
        
        ## 📊 What Makes This Different
        
        ### ✅ **Real Machine Learning**
        - Uses actual trained models (Random Forest, Gradient Boosting, Logistic Regression)
        - Cross-validation for robust performance
        - Feature importance analysis
        
        ### 🎯 **Universal Compatibility**
        - Works with any business dataset
        - Automatic data preprocessing
        - Handles missing values and categorical data
        
        ### 📈 **Actionable Insights**
        - Risk-based recommendations
        - Feature importance rankings
        - Performance metrics and visualizations
        
        ## 🔧 Technical Features
        
        - **Automatic Data Preprocessing**: Handles missing values, encodes categories
        - **Model Selection**: Compares multiple algorithms, selects best performer
        - **Cross-Validation**: Ensures robust, generalizable results
        - **Feature Engineering**: Automatically optimizes input features
        - **Interpretability**: Shows what factors drive churn predictions
        
        ## 🎯 Get Started Now!
        
        1. Prepare your customer data in CSV format
        2. Upload it using the "📤 Upload & Train" tab
        3. Train your custom churn prediction model
        4. Start making predictions and saving customers!
        
        ---
        
        **💡 Pro Tip**: The more relevant customer features you include, the better your predictions will be!
        """)

if __name__ == "__main__":
    main()
