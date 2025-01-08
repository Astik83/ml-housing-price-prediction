


# ML Housing Price Prediction 🏠💸

Welcome to the **ML Housing Price Prediction** project! In this project, we leverage machine learning to predict housing prices based on various features such as area, number of rooms, stories, parking space, and more. This solution applies regression techniques to forecast real estate prices and provides valuable insights for anyone interested in the housing market.

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Data Overview](#data-overview)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)

## 🏡 Project Overview

The objective of this project is to build a predictive model that can forecast the prices of houses based on various input features. This problem is solved using machine learning regression algorithms.

### Key Features in the Dataset:
- **Area (sq. ft.)** 📏
- **Number of Rooms** 🛏️
- **Number of Stories** 🏢
- **Parking Availability** 🚗
- **Furnishing Status** 🛋️
- **Location (prefarea, mainroad)** 🌍
- **Price** 💰 *(Target Variable)*

By training a machine learning model on historical housing data, the goal is to predict house prices for future listings.

## 🛠️ Technologies Used

- **Python** 🐍: The core programming language.
- **Pandas** 📊: Data manipulation and analysis.
- **Scikit-learn** 📚: Machine learning models and tools.
- **Matplotlib** 📈: Data visualization.
- **Seaborn** 🎨: Advanced statistical plotting.
- **Jupyter Notebook** 📝: For interactive development and running code.
- **NumPy** 🔢: For numerical operations.

## 🚀 Getting Started

To get started with this project, follow these simple steps:

### 1. Clone the repository:

```bash
git clone https://github.com/Astik83/ml-housing-price-prediction.git
cd ml-housing-price-prediction
```

### 2. Install the required dependencies:

We have a `requirements.txt` file that lists all the necessary libraries. Install them by running:

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook or Python script:

To train the model and make predictions, you can run the provided Jupyter Notebook (`housing_price_prediction.ipynb`) or the Python script (`housing_price_prediction.py`).

For Jupyter Notebook:

```bash
jupyter notebook housing_price_prediction.ipynb
```

For Python script:

```bash
python housing_price_prediction.py
```

The model will process the data, train the regression model, and display evaluation metrics.

## 📂 Data Overview

The dataset used in this project contains information about houses, such as:

- **Area**: Size of the house in square feet.
- **Stories**: Number of stories in the house.
- **Parking**: Parking space available for the house.
- **Total Rooms**: Number of rooms in the house.
- **Furnishing Status**: Whether the house is furnished.
- **Location**: Categorized into `prefarea` (preferred area) and `mainroad` (near main road).

The **target variable** is the **price** of the house, which the model aims to predict.

### Data Preprocessing:
- **Handling Missing Values**: We handle any missing data by either removing rows or imputing values.
- **Encoding Categorical Variables**: We use `LabelEncoder` to convert categorical features into numeric values.
- **Feature Scaling**: We scale numerical features to standardize the input data for better performance.

## 📊 Model Evaluation

The model's performance is evaluated using:

### 1. **Mean Squared Error (MSE)**:
MSE measures the average squared difference between the actual and predicted values. A lower value indicates a better fit.

### 2. **R-squared (R²)**:
R² indicates how well the model explains the variability of the target variable. A higher R² value (closer to 1) means a better model fit.

## 🖥️ Usage

Once you've set up the project and trained the model, you can:

- **Make Predictions**: Use the trained model to predict the housing price for new input data.
- **Visualize Results**: Visualize the correlation between features and the target variable.
- **Evaluate Model Performance**: Review the evaluation metrics such as MSE and R² to assess the model's effectiveness.

### Example Usage:

After training the model, you can make predictions like so:

```python
# Make a prediction
predicted_price = model.predict([[3000, 3, 1, 2, 0, 1, 1]])  # Example feature values
print(f"Predicted Housing Price: ${predicted_price[0]:,.2f}")
```

## 📈 Visualizations

The project includes visualizations to help understand the relationships between different features and the target variable:

- **Correlation Heatmap**: Visualizes the correlation between features and the price.
- **Feature Importance**: Helps in understanding which features are more important for predicting housing prices.

---

## 📧 Contact  
For queries or suggestions, reach out:  
- **GitHub**: [Astik83](https://github.com/Astik83)  
- **Email**: [shahastik123@gmail.com] 
