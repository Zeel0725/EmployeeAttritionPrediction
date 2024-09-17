# Employee Attrition Prediction Model

## Project Overview
This project aims to predict employee attrition using various machine learning models. The goal is to analyze employee data and create a predictive model to help organizations retain talent and reduce turnover. We used a range of models, including Logistic Regression, Decision Tree, and Random Forest. Additionally, feature engineering was applied to enhance the performance of the Decision Tree and Random Forest models during the second iteration.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset Description](#dataset-description)
- [Modeling Process](#modeling-process)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [How to Use](#how-to-use)
- [License](#license)

## Technologies Used
- Python (scikit-learn, pandas, numpy, matplotlib, seaborn)
- Jupyter Notebook

## Dataset Description
The dataset used for this project contains information about employees, including demographic, job role, satisfaction, and performance metrics.

### Key Features:
- **Age**
- **Department**
- **Job Role**
- **Monthly Income**
- **Job Satisfaction**
- **Years at Company**
- **Attrition (Target)**

## Modeling Process

1. **Data Cleaning and Preprocessing**:  
   The dataset was cleaned by handling missing values and encoding categorical variables. Standardization was applied where necessary.

2. **Initial Model Building**:
   - **Logistic Regression**: A simple baseline model was built to classify employee attrition. It served as a benchmark for comparison with more complex models.
   - **Decision Tree**: A Decision Tree model was built to capture non-linear patterns.
   - **Random Forest**: The Random Forest model was used to improve accuracy by aggregating the output of multiple decision trees.

3. **Feature Engineering**:  
   In the second round, I created new features from the existing data to better capture patterns that affect employee attrition. These features included interaction terms, scaling, and handling class imbalance.

4. **Second Round of Model Building**:
   - **Refined Decision Tree**: After feature engineering, a second Decision Tree model was built to see if the new features improved the performance.
   - **Refined Random Forest**: Similarly, a second iteration of the Random Forest model was built with the new features.

## Results
- **Logistic Regression**: Initial accuracy of X%
- **Decision Tree (1st round)**: Accuracy of Y%
- **Random Forest (1st round)**: Accuracy of Z%
- **Decision Tree (2nd round)**: Accuracy after feature engineering improved to P%
- **Random Forest (2nd round)**: Best performance with an accuracy of Q%

The refined models with feature engineering outperformed the initial models, with the Random Forest showing the highest prediction accuracy.

## Future Enhancements
- Implement other advanced models such as Gradient Boosting or XGBoost.
- Perform hyperparameter tuning to further improve model performance.
- Explore other feature engineering techniques like binning and polynomial features.
- Evaluate the model on a larger, more diverse dataset.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Zeel0725/EmployeeAttritionPrediction.git
   cd EmployeeAttritionPrediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## License
   This project is licensed under the MIT License.

