import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns # For mathematical calculations
from sklearn import preprocessing 
import matplotlib.pyplot as plt  # For plotting graphs
from scipy import stats
import streamlit as st
import time
import pandas as pd
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value
from itertools import product
import numpy as np
class DecisionTreeRegressorr():
   # called every time an object is created from a class
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
       
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split #specifies the minimum number of samples required to split an internal node
        self.max_depth = max_depth  #determines the maximum depth of the decision tree that will be constructed
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        #ensure that the first value you encounter will be greater than the current maximum value
        max_var_red = -float("inf")
        # loop over all the features in the dataset
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            #it will assighn the unique values in the dataset
            possible_thresholds = np.unique(feature_values)
            # loop over all the unique feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null 
                if len(dataset_left)>0 and len(dataset_right)>0:
                   y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                   # compute varience reduction for target variable
                   curr_var_red = self.variance_reduction(y, left_y, right_y)
                   # update the best split if needed
                   # if curr_var_red is greater than max_var_red then it will update the best split as this value
                   if curr_var_red>max_var_red:
                     best_split["feature_index"] = feature_index
                     best_split["threshold"] = threshold
                     best_split["dataset_left"] = dataset_left
                     best_split["dataset_right"] = dataset_right
                     best_split["var_red"] = curr_var_red
                     max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        # if feature value or index is less than or equal to the threshold then the value is assighned to left 
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        # if feature value or index is greater than  to the threshold then the value is assighned to right 
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        # after splitiing it will return the result
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction gfeature'''
        
        weight_l = len(l_child) / len(parent)# calculates the weight of the left child node relative to its parent node
        weight_r = len(r_child) / len(parent)# calculates the weight of the right child node relative to its parent node
        #after calculating weight of left and right child then using this we are going to calculate the varience reduction using this formula 
        #taking sum of the right and left child and subtracting with the varience of parent 
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        #return the varience reduction
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        #calculates the mean value of the target variable Y in a leaf node
        val = np.mean(Y)
        return val  

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        #it will separate the data into independent and dependet 
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        #this dictionory will store best split value
        best_split = {}
        # split until stopping conditions are met
        if (num_samples>=self.min_samples_split) and (curr_depth<=self.max_depth).any():
          # find the best split
          best_split = self.get_best_split(dataset, num_samples, num_features)
          # check  varience reduction
          if best_split["var_red"]>0:
            left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
            #  recursively builds the right subtree of a decision tree node
            right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
            # return decision node
            return Node(best_split["feature_index"], best_split["threshold"],left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    
    #it will shows the how the tree will be build 
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree 
        This fit() function is used to train (fit) a decision tree model using a training dataset
        X and corresponding target variable Y.'''
        

        
        self.root = self.build_tree(X,Y)
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset 
        this function is used to make predictions for new data points x using a trained decision tree model tree'''
        
        if tree.value!=None: return tree.value
        x['feature_name'] = pd.to_numeric(x['feature_name'], errors='coerce')
        x.dropna(inplace=True)


        feature_val = x[tree.feature_index]
        #the feature value is less than or equal to the threshol value of the tree it will make prediction on left tree else make prediction on right tree
        if float(feature_val) <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' function to predict a single data point 
        This function call recursively traverses the decision tree based on the feature values of x, 
        until it reaches a leaf node, at which point it returns the predicted output value'''
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def set_params(self, **params):
      '''function is used to set the values of the attributes of a decision tree object. The function takes a variable 
      number of keyword arguments (**params), 
      which are pairs of attribute names and their corresponding values that should be set for the decision tree object'''
      for param, value in params.items():
        setattr(self, param, value)
        return self

    def r2_score(y_true, y_pred):
      '''function calculates the R-squared score for a set of true target variable values (y_true) 
      and corresponding predicted target variable values (y_pred)'''
      ss_res = np.sum((y_true - y_pred) ** 2)
      #SS_res is the residual sum of squares (the sum of squared differences between the true and predicted values)
      ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
      #SS_tot is the total sum of squares (i.e., the sum of squared differences between the true values and the mean of the true values)
      r2 = 1 - ss_res / ss_tot
      return r2
    
    def mean_squared_error(self,y_true, y_pred):
   
      # Check if the lengths of both arrays are equal
      if len(y_true) != len(y_pred):
          raise ValueError("Length of y_true and y_pred should be the same.")
      
      # Calculate the squared differences between the true and predicted values
      squared_differences = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
      
      # Calculate the mean of the squared differences
      mse = sum(squared_differences) / len(squared_differences)
      
      return mse


# Define a function to load data
@st.cache_data()
def read_csv(file_path):
    return pd.read_csv(file_path)

# Define a function to merge datasets
def merge_datasets(df1, df2):
    # merge the two datasets on a common column
    merged_df = pd.merge(df1, df2, on='common_column')
    return merged_df
# Define a function to scale the dataset
def scale_dataset(df, scaling_type, scaling_columns):
    if scaling_type == 'Standardization':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaled_columns = scaling_columns if scaling_columns else df.columns
    scaled_df = pd.DataFrame(scaler.fit_transform(df[scaled_columns]), columns=scaled_columns)
    df[scaled_columns] = scaled_df
    return df
def remove_outliers(data):
        z_scores = np.abs(stats.zscore(data))
        data_clean = data[(z_scores < 3).all(axis=1)]
        return data_clean

def fill_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.fillna(data.median(), inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.fillna(data.median(), inplace=True)

    if axis == 0:
        return data
    elif axis == 1:
        return data.T



 

def drop_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.dropna(axis=axis, inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.dropna(axis=axis, inplace=True)

    return data


        

# Define the Streamlit app
def app():

    # Add menu items to the Streamlit app's menu
    
    # Set the page title and icon
    st.set_page_config(page_title='DecisionTree Regression App', page_icon=':memo:', layout='wide',initial_sidebar_state="auto")

    
    # Add a sidebar with a menu
    st.sidebar.write("MENU")
    option = st.sidebar.selectbox(
        "Select an option",
        ("Home", "About Decision Tree Regressor", "For more info")
    )

    # Display different content depending on the selected option
    if option == "Home":
        st.write("Welcome to Decision Tree Regressor App!")
    elif option == "About Decision Tree Regressor":
        st.write("""Decision trees are one of the most popular algorithms used for machine learning tasks, especially when it comes to regression problems. In this blog, we'll explore what decision tree regressors are and how they work.
What is a Decision Tree Regressor?
A decision tree regressor is a type of regression algorithm that uses a decision tree as a predictive model. It works by recursively splitting the data into smaller subsets based on the value of a particular feature. Each split divides the data into two or more subsets, and the process is repeated until a stopping criterion is met. This produces a tree-like structure where each internal node represents a decision based on a feature, and each leaf node represents a prediction.
How does it work?
The decision tree regressor starts by selecting the feature that provides the best split. The best split is the one that maximizes the difference between the predicted values of the two subsets created by the split. This process is repeated until a stopping criterion is met, such as when all the data in a subset has the same value or when a maximum depth is reached.
To make a prediction, the decision tree regressor starts at the root node of the tree and follows the path through the tree based on the value of the feature at each internal node. The final prediction is the value of the leaf node that is reached.
Advantages of Decision Tree Regressors
There are several advantages to using decision tree regressors:
Easy to interpret: Decision trees are easy to visualize, and the rules used to make predictions are easy to understand.
Non-parametric: Decision trees don't make any assumptions about the distribution of the data, making them useful for datasets with complex relationships.
Robust: Decision trees can handle missing data, outliers, and noisy data.
Can handle both categorical and numerical data: Decision trees can handle both types of data, making them versatile.
Disadvantages of Decision Tree Regressors
There are also some disadvantages to using decision tree regressors:
Overfitting: Decision trees can be prone to overfitting, which can lead to poor performance on new data.
Instability: Small changes in the data can lead to large changes in the structure of the decision tree.
Bias: Decision trees can be biased towards features with many values or towards features that appear earlier in the tree.
Conclusion
Decision tree regressors are a powerful tool for regression tasks. They are easy to interpret and can handle both categorical and numerical data. However, they can be prone to overfitting and instability, so it's important to tune the hyperparameters and use techniques like pruning to prevent these issues. Overall, decision tree regressors are a useful addition to any data scientists toolkit.""")
    elif option == "For more info":
        st.write("follow this link : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html")

    # Set the app title
    st.title('DecisionTree Regression App')
        
    # Allow users to upload two datasets
    st.sidebar.header('Upload datasets')
    Choose_file  = st.sidebar.selectbox("Select filfe upload type", ("uplode one file", "uplode Two files",))
    if Choose_file== "uplode one file":
        uploaded_file = st.sidebar.file_uploader('Upload dataset ', type=['csv', 'xlsx'])
        if uploaded_file is not None:
            with st.spinner('Loading datasets...'):
                merged_df = pd.read_csv(uploaded_file,na_values=['?', '/', '#',''])
                
            
                st.write('Merged Dataset')
                
                st.write(merged_df)
        
    elif Choose_file =='uplode Two files':

        # Upload the first dataset
        uploaded_file1 = st.sidebar.file_uploader('Upload dataset 1', type=['csv', 'xlsx'])

        # Upload the second dataset
        uploaded_file2 = st.sidebar.file_uploader('Upload dataset 2', type=['csv', 'xlsx'])

         # Merge the two datasets
        if uploaded_file1 is not None and uploaded_file2 is not None:
            with st.spinner('Loading datasets...'):
                df1 =  pd.read_csv(uploaded_file1,na_values=['?', '/', '#',''])
                df2 =  pd.read_csv(uploaded_file2,na_values=['?', '/', '#',''])
            with st.spinner('Merging datasets...'):
                merged_df = pd.merge(df1, df2, on='id')
                st.write('Merged Dataset')
                
                
       
            # show entire data
        if st.sidebar.checkbox("Show all data"):
            st.write(merged_df)

        st.subheader('To Check Columns Name')
        # show column names
        if st.sidebar.checkbox("Show Column Names"):
            st.write(merged_df.columns)

        # show dimensions
        if st.sidebar.checkbox("Show Dimensions"):
            st.write(merged_df.shape)

        st.subheader('Summaery of the Data')     
        # show summary
        if st.sidebar.checkbox("Show Summary"):
            st.write(merged_df.describe())
        st.subheader('Correlation')     
        # show summary
        if st.sidebar.checkbox('Correlation'):
             st.write(merged_df.corr())
  


        


            # Print numerical and categorical features
        st.header('Numerical and Categorical Features')
        numerical_features = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = merged_df.select_dtypes(include=['object']).columns.tolist()
        st.write('Numerical Features')
        st.write(numerical_features)
        st.write('Categorical Features')
        st.write(categorical_features)

        # Perform exploratory data analysis
        st.sidebar.title("Exploratory Data Analysis")
        # get user input for the column to plot
        plot_column = st.sidebar.selectbox('Select a column to plot(HIATOGRAM,COUNTPLOT)', merged_df.columns)
            
        # plot a histogram of the selected column
        plt.figure()
        sns.histplot(data=merged_df, x=plot_column)
        st.pyplot()
            
        # plot a bar chart of the selected column
        if merged_df[plot_column].dtype == 'object':
                plt.figure()
                sns.countplot(data=merged_df, x=plot_column)
                st.pyplot()
        # plot a scatter plot of two numerical columns
        elif merged_df[plot_column].dtype in ['float64', 'int64']:
                plt.figure()
                numerical_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
                x_column = st.sidebar.selectbox('Select a column for X-axis[SCATER PLOT]', numerical_columns)
                y_column = st.sidebar.selectbox('Select a column for Y-axis[SCATER PLOT]', numerical_columns)
                sns.scatterplot(data=merged_df, x=x_column, y=y_column)
                st.pyplot() 
                                                

            
            
        st.sidebar.subheader("Select column for checking target variable distribution with heatmap")
        target_col = st.sidebar.selectbox("Target variable", merged_df.columns)
        if target_col != 'id':
            st.write("Target variable:", target_col)
            st.write("Summary statistics:")
            st.write(merged_df[target_col].describe())
            st.write("Distribution:")
            sns.histplot(data=merged_df, x=target_col, kde=True)
            st.pyplot()
            st.write("Correlation with other variables:")
            corr = merged_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            st.pyplot()
                

        # Show missing values
        st.write('## Missing Values')
        st.sidebar.header('Missing Value Treatment')
        if st.sidebar.checkbox('Show missing values'):
            st.write(merged_df.isnull().sum())

        # Numerical columns
        numerical_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Categorical columns
        categorical_cols = merged_df.select_dtypes(include=['object']).columns.tolist()

        # Fill missing values in numerical columns
        if st.sidebar.checkbox('Checking missing values for numeric columns'):

            num_missing = merged_df.select_dtypes(include=np.number).isna().sum()
            num_missing_percentage = (num_missing / merged_df.shape[0]) * 100
            st.subheader('Missing Values for Numerical Variables')
            st.write(pd.concat([num_missing.rename('Total Missing'), num_missing_percentage.rename('Percentage')], axis=1))
        num_impute = st.sidebar.selectbox('Select a numerical imputation method', 
                                        ('Drop missing values', 'Mean imputation', 'Median imputation', 'Mode imputation'))
        if num_impute == 'Drop missing values':
            df_num = merged_df[numerical_cols].dropna()
        elif num_impute == 'Mean imputation':
            df_num = merged_df[numerical_cols].fillna(merged_df[numerical_cols].mean())
            st.write("Mean value of column is :", df_num)
        elif num_impute == 'Median imputation':
            df_num = merged_df[numerical_cols].fillna(merged_df[numerical_cols].median())
            st.write("Median value of column is :", df_num)
        else:
            df_num = merged_df[numerical_cols].fillna(merged_df[numerical_cols].mode().iloc[0])
            st.write("Mode value of column is :", df_num)

        
        if st.sidebar.checkbox('Checking missing values for Categorical  columns'):

            # Display missing values for categorical variables
            cat_missing = merged_df.select_dtypes(include='object').isna().sum()
            cat_missing_percentage = (cat_missing / merged_df.shape[0]) * 100
            st.subheader('Missing Values for Categorical Variables')
            st.write(pd.concat([cat_missing.rename('Total Missing'), cat_missing_percentage.rename('Percentage')], axis=1))

        # Fill missing values in categorical columns
        cat_impute = st.sidebar.selectbox('Select a categorical imputation method', 
                                        ('Drop missing values', 'Most frequent imputation'))
        if cat_impute == 'Drop missing values':
            df_cat = merged_df[categorical_cols].dropna()
            st.write("Mode value of column is :", df_cat)
        else:
            df_cat = merged_df[categorical_cols].fillna(merged_df[categorical_cols].mode().iloc[0])
            st.write("Mode value of column is :", df_cat)

        # Combine numerical and categorical columns
        merged_df = pd.concat([df_num, df_cat], axis=1)



        if st.sidebar.checkbox("Show Missing   Values after fill"):
            st.write(merged_df.isna().sum()) 
            # To change datatype of a column in a dataframe
            # display datatypes of all columns
        st.sidebar.header('Encoding')
        if st.sidebar.checkbox("Show datatypes of the columns"):
            st.write(merged_df.dtypes)

        st.sidebar.subheader('Convert Datatype')
        # Automatic detection of categorical columns
        cat_cols = merged_df.select_dtypes(include=['object']).columns.tolist()

        # Show categorical columns
        st.write('## Categorical Columns')
        st.write(cat_cols)

        # Encoding options
        encode_option = st.sidebar.selectbox('Select an encoding method', ('None', 'Label Encoding', 'One-Hot Encoding'))

        # Apply encoding method
        if encode_option == 'Label Encoding':
            le = LabelEncoder()
            merged_df[cat_cols] = merged_df[cat_cols].apply(le.fit_transform)
        elif encode_option == 'One-Hot Encoding':
            merged_df = pd.get_dummies(merged_df, columns=cat_cols)

        

                        


        if st.sidebar.checkbox("Show updated datatypes of the columns"):
            st.write(merged_df.dtypes)

        if st.sidebar.checkbox("Preview Dataset aftre convert datatype"):
             st.write(merged_df.head())
                        

                        

        # Display outliers in boxplot
        st.header("Display Outliers")
        num_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        col = st.selectbox("Select a numerical column to display outliers", num_cols)
        if col is not None:
                fig, ax = plt.subplots()
                ax.boxplot(merged_df[col], vert=False)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)

        st.sidebar.header('Outlier Treatment')
        show_outliers = st.sidebar.checkbox("Show outliers")

        # Display data with or without outliers
        if show_outliers:
            for k, v in merged_df.items():

                q1 = v.quantile(0.25)
                q3 = v.quantile(0.75)
                irq = q3 - q1
                v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
                perc = np.shape(v_col)[0] * 100.0 / np.shape(merged_df)[0]
                print("Column %s outliers = %.2f%%" % (k, perc))
                st.write(k,perc)


        method = st.sidebar.selectbox("Select outlier detection method", ("NONE","IQR", "Z-score"))

        if st.sidebar.checkbox("Fill Outliers"):
            if method == "IQR":
                merged_df = fill_outliers(merged_df, method='iqr', axis=0)
            elif method == "Z-score":
                merged_df = fill_outliers(merged_df, method='zscore', axis=0)

                st.write("Data with filled outliers")
                st.write(merged_df)

        if st.sidebar.checkbox("Drop Outliers"):
            if method == "IQR":
                merged_df = drop_outliers(merged_df, method='iqr', axis=0)
            elif method == "Z-score":
                merged_df = drop_outliers(merged_df, method='zscore', axis=0)

            st.write("Data with dropped outliers")
            st.write(merged_df)




        show_outliers = st.checkbox("Show outliers aftre treatement")

        # Display data with or without outliers
        if show_outliers:
            for k, v in merged_df.items():
                    
                    q1 = v.quantile(0.25)
                    q3 = v.quantile(0.75)
                    irq = q3 - q1
                    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
                    perc = np.shape(v_col)[0] * 100.0 / np.shape(merged_df)[0]
                    print("Column %s outliers = %.2f%%" % (k, perc))
                    st.write(k,perc)

        

       # Add a button to start feature scaling
        st.sidebar.header(" Feature Scaling")
        # Create feature selection options
        feature_options = list(merged_df.columns)
        scaling_columns = st.sidebar.multiselect('Select columns to scale', feature_options)

        # Create scaling options
        scaling_options = ['None','Standardization', 'Normalization']
        scaling_type = st.sidebar.selectbox('Choose a scaling method', scaling_options)

            # Scale the merged dataset
        merged_df = scale_dataset(merged_df, scaling_type, scaling_columns)
        st.write('Scaled Dataset')
        st.write(merged_df)


        with st.spinner('Splitting dataset into training and testing sets...'):

            # Allow users to select the target variable and the features for the model
            st.header('Select target variable and features')
            target_col = st.selectbox('Select the target variable', options=merged_df.columns)
            feature_cols = st.multiselect('Select the features', options=merged_df.columns.drop(target_col))

            # Perform train_test_split
            st.header('Train-test split')
            test_size = st.slider('Select the test size', min_value=0.1, max_value=0.5, step=0.1, value=0.2)
            X_train, X_test, y_train, y_test = train_test_split(merged_df[feature_cols], merged_df[target_col], test_size=test_size, random_state=0)
            # train the decision tree regression model
        with st.spinner('Training the model...'):
            st.header('Model Building')
            alg = ['Decision Tree Regressor','Random Forest Regressor','Linear regression']
            classifier = st.selectbox('Choose the Algorithem', alg)
            if classifier=='Decision Tree Regressor':
                # Build the DecisionTree regression model
                
                max_depth = st.slider('Select the maximum depth', min_value=1, max_value=20, value=3)
                min_samples_split=st.slider('Select the minimum sample split', min_value=1, max_value=20, value=3)
                model = DecisionTreeRegressor(max_depth=max_depth,min_samples_split=min_samples_split)
                model.fit(X_train, y_train)
                st.write('Decision Tree Regressor Using Sk-learn')
                # Evaluate the model performance
                st.header('Model performance')
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write('MSE:', mse)
                st.write('R2 score:', r2)
                    
           
            elif classifier == 'Random Forest Regressor':
                n_estimators=st.slider('Select the maximum depth', min_value=10, max_value=100, value=20)
                max_depth = st.slider('Select the maximum depth', min_value=1, max_value=20, value=3)
                min_samples_split=st.slider('Select the minimum sample split', min_value=1, max_value=20, value=3)
                from sklearn.ensemble import RandomForestRegressor
                RFC=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
                RFC.fit(X_train, y_train)
                st.header('Model performance')
                y_pred = RFC.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write('MSE:', mse)
                st.write('R2 score:', r2)
            elif classifier == 'Linear regression':
                from sklearn.linear_model import LinearRegression
                
                LR=LinearRegression()
                LR.fit(X_train, y_train)
                st.header('Model performance')
                y_pred = LR.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write('MSE:', mse)
                st.write('R2 score:', r2)
          

                  

    'Starting a long computation...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(10):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    '...and now we\'re done!'

if __name__ == '__main__':
        app()