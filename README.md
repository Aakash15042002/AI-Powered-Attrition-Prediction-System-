# AI-Powered Attrition Prediction System 

## OBJECTIVE: 
Build a simple AI system that predicts whether an employee is likely to leave the company 
(attrition). 

This task assesses your ability to: 

* Clean and preprocess real
* world data - Build a basic ML model
* Evaluate performance
* Document findings and communicate clearly 
## VARIABLES

* Age: Represents the age of the employee in years. Age can correlate with experience, job satisfaction, and retention.

* Attrition: A categorical feature that indicates whether the employee has left the company ("Yes" or "No"). This is typically the target variable in attrition or employee turnover analysis.

* BusinessTravel: Categorizes how frequently the employee travels for business. Possible categories might include "Rarely," "Frequently," and "Non-Travel." Frequent business travel may impact job satisfaction and attrition.

* DailyRate: The daily wage of the employee. This feature can be used to analyze the financial aspect of employee satisfaction.

* Department: Categorical feature representing the department the employee works in, such as "Sales," "R&D," or "HR." Departmental differences can impact work environment and engagement.

* DistanceFromHome: Indicates the distance between the employee's home and workplace, measured in kilometers or miles. Longer commutes might be linked to job dissatisfaction and turnover.

* Education: Educational level attained by the employee, typically represented by an integer (e.g., 1 for high school, 2 for bachelor’s, etc.). Education level could correlate with salary, job role, and career progression.

* EducationField: The field of study in which the employee received their education (e.g., "Life Sciences," "Marketing"). This could influence job role suitability and satisfaction.

* EmployeeCount: A constant feature in this dataset, with the value set to 1 for all employees. It may not have predictive value but is included for completeness.

* EmployeeNumber: A unique identifier for each employee. This is a non-predictive feature used for record-keeping.

* EnvironmentSatisfaction: Measures employee satisfaction with their work environment on a Likert scale (1-4). Higher values indicate greater satisfaction, which may correlate with job performance and retention.

* Gender: Categorical feature indicating the employee’s gender. Gender may be used to analyze diversity, inclusion, and disparities in areas such as pay or promotion.

* HourlyRate: The hourly wage rate for the employee. Like DailyRate, it provides insight into the financial compensation received by the employee.

* JobInvolvement: Reflects the degree to which the employee is involved in their job on a Likert scale (1-4). Higher values indicate greater involvement, which may correlate with performance and retention.

* JobLevel: Represents the level or rank of the employee within the organization (e.g., entry-level, manager, senior manager). Job level may affect salary, responsibilities, and attrition.

* JobRole: Categorical feature denoting the employee's specific job role (e.g., "Sales Executive," "Manager"). This feature can provide insights into job fit and satisfaction.

* JobSatisfaction: Measures overall job satisfaction on a Likert scale (1-4). Higher values suggest higher satisfaction and lower attrition risk.

* MaritalStatus: The marital status of the employee (e.g., "Single," "Married," "Divorced"). Marital status may influence work-life balance and job engagement.

* MonthlyIncome: Represents the monthly salary of the employee. Income can directly affect employee satisfaction and retention.

* MonthlyRate: Similar to MonthlyIncome but calculated as a rate. This feature may provide an additional layer of financial analysis for the employee.

* NumCompaniesWorked: Indicates the number of companies the employee has worked for before their current job. This could be a predictor of job stability or risk of attrition.

* Over18: A categorical feature indicating whether the employee is over 18 years old. Since all employees are likely to be over 18, this feature may not provide any useful variation.

* OverTime: A categorical feature indicating whether the employee works overtime. Employees working frequent overtime may experience higher stress, impacting job satisfaction and attrition.

* PercentSalaryHike: The percentage increase in salary for the employee over the last year. This feature may influence retention and job satisfaction.

* PerformanceRating: An integer rating (likely 1-4) representing the employee’s performance. Higher values suggest better performance, which may be linked to higher pay or job retention.

* RelationshipSatisfaction: Measures satisfaction with workplace relationships on a Likert scale (1-4). Higher values suggest better relationships, which can impact job satisfaction and performance.

* StandardHours: The number of hours the employee is expected to work, typically standardized across all employees. This feature may be constant and not particularly predictive.

* StockOptionLevel: Indicates the level of stock options granted to the employee, which can influence financial incentives and retention.

* TotalWorkingYears: The total number of years the employee has been working, including their tenure at the current company. More experience may correlate with higher salary and lower attrition risk.

* TrainingTimesLastYear: Represents the number of training sessions the employee attended last year. Frequent training may be linked to skill development and job satisfaction.

* WorkLifeBalance: Measures the employee's satisfaction with their work-life balance on a Likert scale (1-4). Higher values indicate better balance, which may reduce attrition risk.

* YearsAtCompany: The number of years the employee has worked at the current company. Longer tenures may be linked to higher loyalty and lower attrition.

* YearsInCurrentRole: The number of years the employee has held their current position. Longevity in a role may correlate with job mastery and satisfaction.

* YearsSinceLastPromotion: Represents the number of years since the employee's last promotion. Longer times since promotion may impact job satisfaction and retention.

YearsWithCurrManager: Indicates the number of years the employee has worked with their current manager. Strong relationships with managers may positively affect job satisfaction and performance.

## DATA LOADING 

The first phase of any machine learning project involves loading the data. This step is crucial as it allows us to access the data and prepare it for subsequent processes, including analysis and modelling. Proper data loading sets the foundation for effective data processing, exploration, and ultimately, accurate model training and inference.

## DATA PRE-PROCESSING

In the previous section, we focused on understanding the structure of our dataset and categorizing the features. With this foundational knowledge, we can now move on to data preprocessing. This step involves applying essential techniques such as handling missing values by either removing or imputing them, checking and eliminating duplicate entries, converting categorical data into numerical formats, and removing irrelevant features. These preprocessing steps are critical to ensure that the data is clean, consistent, and ready for effective model training.


## Here is a brief description of the categorical features:

* Attrition: This is the target variable, indicating whether an employee has left the company ("Yes") or stayed ("No"). It’s binary and essential for predicting employee turnover.

* BusinessTravel: Categorizes employees based on the frequency of business travel. Employees who travel more frequently may experience different job satisfaction and work-life balance compared to those who rarely travel or don’t travel at all.

* Department: Indicates the department to which employees belong, such as "Sales," "Research & Development," and "Human Resources." Different departments can have varying work cultures, expectations, and job satisfaction levels, which can influence attrition rates.

* EducationField: Represents the educational background of employees, ranging from "Life Sciences" to "Human Resources." The diversity of education fields can affect the roles employees are suited for and their career trajectories.

* Gender: A binary feature representing gender ("Female" or "Male"). Gender may impact how employees experience workplace culture, compensation, and opportunities, though its predictive power should be evaluated carefully to avoid biases.

* JobRole: Represents the specific job function an employee performs, such as "Sales Executive" or "Research Scientist." Job roles may correlate with levels of stress, satisfaction, and opportunities for advancement, influencing retention.

* MaritalStatus: Identifies the employee's marital status as "Single," "Married," or "Divorced." Marital status may affect work-life balance and engagement at work.

* Over18: A constant feature indicating that all employees are over 18 years old ("Y"). Since all values are the same, this feature does not add any variability and may not be useful for predictive modeling.

* OverTime: Indicates whether the employee works overtime ("Yes" or "No"). Overtime work can be associated with job stress and burnout, which might impact job satisfaction and retention.

## Key Considerations:
* Preprocessing: These categorical features will need to be encoded (e.g., one-hot encoding or label encoding) for use in machine learning models.
* Predictive Power: While some of these features may have a strong relationship with attrition (e.g., BusinessTravel, OverTime), others like Over18 may not contribute to the predictive power of the model.
* Bias & Fairness: Care should be taken to ensure that sensitive features such as gender or marital status do not introduce unfair biases into the model predictions.



## Categorizing Discrete and Continuous Variables

* Discrete Variables: Discrete variables are those that take on a limited number of distinct values. They are typically integer-based and represent countable, distinct quantities. A simple yet practical way to identify discrete variables is to look at the number of unique values they have. If a feature has 10 or fewer unique values, it can often be categorized as discrete. While this method may not always be the most rigorous, it generally works well in practice for quick identification. Discrete variables often include features like the number of children, number of promotions, or even categories that have been encoded as integers.

* Continuous Variables: Continuous variables, on the other hand, are those that can take any value within a range and are typically represented as floating-point numbers. These variables usually measure quantities such as age, income, or time spent on a task. Unlike discrete variables, continuous variables have an infinite number of possible values between any two points and are often used to represent real-world measurements.

## Importance of Distinguishing Between Discrete and Continuous Variables

The separation of numerical features into discrete and continuous categories is crucial because each type requires different preprocessing steps. These differences in processing ensure that the features are compatible with machine learning models and allow the models to accurately capture the relationships within the data.

* Discrete Variables: Discrete variables are often treated similarly to categorical variables, even though they are numeric. Processing steps for discrete variables might include normalization or scaling, but they may also involve binning or encoding techniques depending on the model being used. For example, when using tree-based models like Random Forest or XGBoost, discrete variables can be left in their raw integer form, as these models handle them well. However, for linear models or neural networks, it may be beneficial to normalize them.

* Continuous Variables: Continuous variables typically require more advanced preprocessing to ensure they are scaled appropriately. Techniques like min-max scaling or standardization (z-score normalization) are commonly applied to continuous variables. These transformations help ensure that continuous features are on a similar scale, which can prevent certain features from dominating the learning process in gradient-based models like logistic regression or neural networks.

## Intuitive Understanding and Processing

Beyond the technical processing, understanding the distinction between discrete and continuous variables is important from an intuitive perspective as well. Discrete variables represent countable, distinct categories or quantities, which can often provide categorical insights into the data. Continuous variables, however, represent measurable phenomena and can give more nuanced, granular information. Understanding the nature of these variables helps guide the feature engineering process, ensuring that we handle them in ways that enhance the model's performance and predictive power.

For instance, discrete variables like the number of promotions or education level can influence employee attrition in ways that are qualitatively different from continuous variables like monthly income or years at the company. Discrete variables can reflect key milestones or specific categories in an employee's career, whereas continuous variables track the progression of a more gradual, ongoing process.

## Categorical to Numerical Mapping

When converting categorical variables to numerical counterparts in machine learning, a variety of encoding techniques can be employed. These methods include one-hot encoding, binary encoding, ordinal encoding, sine and cosine encoders, M-estimates mean encoding, and more. Each encoding technique has specific use cases and requirements depending on the nature of the feature. The choice of encoding method can significantly influence the predictive power of a machine learning model, which is why selecting the appropriate method for each feature is crucial.

Let’s break down the reasoning behind selecting encoding techniques for some key features in this dataset:

* Attrition (Target Variable): Since the Attrition feature is our target variable, representing whether an employee left the company or stayed, a simple binary encoding can be applied. In this case, we would use 0 to represent "No" (did not leave) and 1 to represent "Yes" (left the company). Binary encoding is appropriate because this feature is a binary categorical variable, and this encoding will maintain its simplicity while serving the needs of our classification model.

* Business Travel: The BusinessTravel feature provides insights into how frequently an employee travels for work. Employees who travel more frequently might experience different levels of job satisfaction, and this could influence their decision to stay or leave the company. There is evidence that travel may correlate with employee well-being—frequent business trips might be seen as a perk by some employees, while others might find it burdensome. Therefore, this feature can be encoded using ordinal encoding or target-based encoding, as it contains a natural ordering (e.g., "Non-Travel," "Travel_Rarely," "Travel_Frequently"). This approach captures the relationships between travel frequency and potential employee attrition while preserving the categorical feature's information.

* Department: The Department feature categorizes employees based on their work area, such as "Sales," "Research & Development," and "Human Resources." Different departments often have unique work cultures, goals, and pressures, which could influence employee attrition. Because there is no inherent order among the departments, one-hot encoding might be the best choice here, as it treats each department as a unique and independent category. One-hot encoding allows us to represent the categorical feature in a way that prevents the model from assuming any ordinal relationships between departments.

* Education Field: The EducationField feature indicates the field of study employees pursued, such as "Life Sciences," "Medical," "Marketing," etc. Education may influence career paths, salary expectations, and job satisfaction. This feature does not have a natural ordering, so one-hot encoding could be effective in representing it. However, if we find that certain fields of education correlate more strongly with attrition, we could also consider using target-based encodings like mean encoding or M-estimate encoding. These techniques will help capture the feature's influence on attrition while reducing dimensionality, especially if the number of categories is large.

* Gender: Although Gender should not ideally impact employee attrition due to ethical and legal considerations, gender bias may still exist in industries. Therefore, it is possible that this feature might have some influence on attrition, but it must be handled carefully to avoid bias in the model. As Gender is binary, we can use simple binary encoding, treating "Male" as 0 and "Female" as 1. However, we must be mindful of how this feature is used and consider fairness metrics in our final model evaluation.

* Job Role: The JobRole feature describes the specific job functions employees perform, ranging from "Sales Executive" to "Research Scientist" and more. Different job roles may have different expectations, pressures, and career trajectories, which could significantly impact employee attrition. Since job roles do not have a natural ordering, one-hot encoding would be a suitable choice. Alternatively, we could apply target-based encoding methods if certain roles have a stronger correlation with attrition.

* Marital Status: The MaritalStatus feature captures whether an employee is "Single," "Married," or "Divorced." While it may seem unrelated to attrition, marital status could influence job satisfaction and work-life balance. For instance, employees with families may prioritize stability, while single employees may be more willing to take risks. This feature could be encoded using one-hot encoding or binary encoding. Further analysis could reveal whether marital status has a meaningful impact on attrition.

* Over18: The Over18 feature is redundant in this case, as it contains only one unique value ("Y") for all employees. Since this feature does not provide any variation or useful information, it can be safely dropped from the dataset.

* Overtime: The OverTime feature indicates whether employees are working overtime. Overtime work can contribute to burnout and dissatisfaction, which may drive employees to leave the company. This binary feature ("Yes" or "No") can be encoded using simple binary encoding. This is an important feature to retain as it could have a direct influence on attrition.

## Statistical Analysis
A key advantage of our dataset is that all features are non-numerical, enabling comprehensive statistical analysis across the entire dataset. This approach provides a thorough understanding of the data. However, it’s important to recognize that some features originally categorical have been converted to numerical formats, which may not reflect their original categorical nature. Therefore, careful attention should be given to the values and features we are analyzing to ensure accurate interpretation and analysis.

# Attrition
Before initiating feature analysis, it is crucial to first examine the target variable. Any imbalance within the target variable can significantly influence both the analysis and the model's performance. Addressing this imbalance early on ensures a more accurate understanding of the data and leads to more reliable modeling outcomes.

Based on the bar graph and pie chart for the Attrition feature, here are the key insights:

* Imbalanced Dataset: The bar graph clearly shows that the majority of the dataset consists of employees who have not experienced attrition (No category). Out of the total, 1,233 employees have stayed with the company, while only 237 employees have left (Yes category). This creates a significant imbalance between the two categories.

* Attrition Rate: The pie chart provides a visual representation of the attrition rate within the company. Only 16.1% of the employees have experienced attrition, while 83.9% remain with the company. This indicates that the company has a relatively low attrition rate, which could suggest good employee retention overall.

* Implications for Analysis: The significant imbalance between the No and Yes categories in the dataset should be taken into consideration when performing further analysis, such as predictive modeling. Techniques such as resampling (oversampling the minority class or undersampling the majority class) or using algorithms designed to handle imbalanced data (e.g., SMOTE, ADASYN) may be necessary to ensure that predictive models do not become biased towards the majority class.

* Strategic Focus: Although the attrition rate is relatively low, it is still essential for the company to investigate the reasons behind the 16.1% attrition. Understanding the factors contributing to employee turnover can help in refining retention strategies, improving job satisfaction, and ensuring that high-performing employees remain with the company.

This analysis provides a clear overview of the attrition dynamics within the dataset, highlighting the need to consider the class imbalance in future analyses and to focus on strategies that can further reduce employee turnover.

In the dataset, there is a significant class imbalance, with 1,233 instances of employees who remain with the company compared to only 237 instances of those who leave. This imbalance may skew the results of any analysis and negatively impact the model's overall performance, making it difficult to accurately predict attrition. Given this disparity, drawing precise conclusions from the minority class may not be entirely reliable.

One potential approach to address this imbalance is to implement under-sampling of the majority class. By reducing the number of samples in the majority class, the model can be trained to give more attention to the minority class. This method ensures that the model does not become overly biased towards the majority class and instead generalizes better across both classes. While under-sampling may result in a loss of some data from the majority class, it can help create a more balanced and robust model that effectively addresses the issue of class imbalance.


In this scenario, the no-information rate, or the null error rate, for the entire dataset is 16.12%. This rate persists at around 16% even when calculated separately for training subsets. This finding is concerning as it implies that a model predicting only the dominant class (i.e., employees who do not leave the company) would exhibit a relatively low error rate without effectively addressing the minority class (employees who do leave the company).
Such an outcome suggests that the model may become overly biased towards predicting the majority class, thus undermining its ability to accurately identify and focus on the critical minority class, which is of primary interest. Specifically, the goal is to understand the factors leading to employee attrition and to accurately predict when and why employees leave the company. If the model continues to lean heavily toward the "no attrition" class, it risks missing crucial insights into the "attrition" class, thereby limiting its utility in addressing the core issue of employee turnover.

# Categorical Features
Categorical features provide valuable insights into the qualitative aspects of the data. To gain a comprehensive understanding, it is prudent to first analyze these categorical variables. This analysis will establish a foundation for subsequent examination of numerical features, allowing us to explore their relationships in the context of the categorical variables. This approach will offer a more nuanced and thorough understanding of the dataset.
The distribution of business travel among employees shows that the majority, approximately 71%, engage in minimal travel, indicating that business travel is not a common activity across the entire workforce. This aligns with typical organizational practices where extensive business travel is generally reserved for specific roles.

A smaller subset of employees, around 18%, travel frequently. This group is likely composed of individuals in specialized or senior roles that require regular interaction with clients, partners, or other business locations, reflecting the selective nature of business travel in the company.

Additionally, about 10% of employees do not engage in any business travel at all. These employees may occupy roles that are primarily office-based or focused on internal operations, where travel is unnecessary.

This distribution underscores that business travel within the organization is not a widespread requirement and is typically limited to a select group of employees who likely have responsibilities that necessitate travel. The majority of employees have minimal travel obligations, highlighting the role-specific nature of business travel in the company.

The dataset reveals that the company is heavily focused on its Research and Development (R&D) department, which employs 961 individuals, making up approximately 65.4% of the total workforce. This suggests that the company places a significant emphasis on innovation and product development, which is common in the tech industry. A strong R&D presence often indicates that the company prioritises developing new products and maintaining a competitive edge in the market through continuous innovation.

The Sales department, with 446 employees (around 30.3% of the workforce), also plays a crucial role, though it is smaller in comparison to R&D. This distribution reflects the importance of marketing and selling the company's products, but it highlights that the company’s primary focus remains on developing those products.

The Human Resources (HR) department, with only 63 employees (about 4.3% of the workforce), is the smallest department. This suggests that while HR is essential for supporting the workforce, the company allocates fewer resources to this area compared to R&D and Sales. This is not uncommon in tech companies where the focus is more on technical expertise and product development.

Overall, this distribution indicates that the company, likely within the tech industry (as suggested by IBM’s presence), prioritises innovation and R&D over other functions. A strong emphasis on R&D is crucial for staying competitive in a rapidly evolving market. However, a more modest focus on Sales and HR might imply that the company values technical advancement and innovation above other operational areas, though both are still vital for the company’s overall success.

The graph shows the distribution of employees across different departments, segmented by attrition (Yes/No). While there are apparent differences in attrition rates among departments, these variations are likely influenced by the imbalanced representation of employees across the departments. Here's a more detailed interpretation:

* Class Imbalance: The Research and Development (R&D) department has the largest number of employees (961), followed by Sales (446), and finally Human Resources (63). Given this significant imbalance in the size of the departments, the attrition counts should be interpreted in context. For instance, while R&D has the highest attrition count (133 employees), it also has the largest workforce, so the proportion of attrition may not be as high as it seems at first glance.

* Proportionate Attrition: Although the absolute numbers of attrition are higher in R&D and Sales, it is important to look at the attrition rate relative to the total number of employees in each department. Simply comparing raw numbers might mislead the interpretation of which department experiences more attrition.

* Caution in Interpretation: The graph's results should be interpreted with caution due to these imbalances. Without normalizing the data (e.g., calculating attrition rates as a percentage of each department's total employees), it is difficult to draw any firm conclusions about a relationship between department and attrition. The observed differences may primarily reflect the overall distribution of employees across departments rather than any meaningful pattern related to attrition.

* Further Analysis Required: To gain a clearer understanding of the relationship between department and attrition, it would be useful to calculate and compare attrition rates (i.e., the percentage of employees leaving each department). This could reveal whether certain departments are experiencing disproportionately high attrition relative to their size, which could warrant further investigation.

Now probably a relationship can be seen it turns out that those who travel frequently are the highest in research and Development Department and this is quite normal because those who seems to travel frequently are must be belonging to the research and development they are traveling to do some kind of research or some kind of identify new technologies and understand the trends and all that whereas for sales only 84 individuals travel frequently and for human resources only eleven travel frequently it's got interesting to see that even in human resources there are some individuals who are traveling frequently there are also some individuals who travel very rarely so that is somebody like and some individuals who do not travel at all but a travel frequently shows a quite interesting pattern as it's quite huge for research and development department as compared to sales and humans resources

Key Observations:

** R&D Department: While R&D has the highest total number of employees, the majority travel infrequently. This suggests that R&D roles are primarily focused on activities that do not require frequent business travel.
** Sales Department: Sales shows a more varied distribution of travel frequency, with a significant portion of employees traveling rarely or frequently, depending on their specific roles. This is typical of sales teams, where some positions require regular travel to meet with clients, while others may be more office-based.
** HR Department: The HR department shows a minimal need for travel overall, with only a small number of employees traveling frequently. This reflects the nature of HR work, which is often centralized, but certain roles may require travel for recruitment, training, or company-wide initiatives.

The analysis of the pie chart and bar graph suggests that the feature related to educational qualifications may not be highly significant for understanding employee distribution within the company. This feature includes six distinct categories: Unambiguous, Life Science, Medical, Marketing, Technical Degree, and Human Resources. The distribution across these categories reveals the following:

Human Resources: This category has the smallest representation with only 27 employees, which is consistent with the overall low number of employees in the Human Resources department, totaling 63. This implies that the remaining employees in the Human Resources department likely come from other educational backgrounds.

Technical Degree: There are 159 employees with a technical degree. Although the exact distribution among departments is not specified, it is reasonable to infer that many of these employees are likely part of the Research and Development (R&D) department, given its focus on technical expertise.

Marketing: Employees with marketing qualifications total 464. This aligns with the observation that the Sales department has a significant number of employees, as marketing qualifications are typically relevant for sales-related roles.

Medical: With 606 employees holding medical degrees, this category supports the earlier finding that the R&D department also includes a substantial number of employees with medical backgrounds, as this department often conducts research in life sciences and related fields.

Life Science: This category dominates with 606 employees, corroborating the prominence of the R&D department, which focuses heavily on life sciences research.

Overall, there appears to be a correlation between the educational qualifications and departmental affiliation. Specifically, the R&D department accommodates a large number of employees with life science and medical qualifications, while the Sales department includes many with marketing backgrounds. The technical degree holders are likely concentrated in the R&D department as well. The small number of employees with human resources education relative to the size of the Human Resources department suggests that this educational background is less common among the department's employees, with others potentially working in Sales or other departments.

This analysis indicates that while educational qualifications provide some insight into departmental distribution, the most significant trends are observed in the alignment of life science and medical qualifications with R&D, and marketing qualifications with Sales. The specific role of technical degrees and the distribution of other educational backgrounds within departments warrant further investigation.

The dataset reveals notable variations among the different job roles within IBM, encompassing a total of nine distinct roles: Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, and Human Resources.

Human Resources: There are 52 employees identified with the Human Resources job role. This is in contrast to the 27 employees with a Human Resources education and 63 employees within the Human Resources department, indicating a potential discrepancy in role allocation or a different distribution of job titles within the department.

Sales Executive: The largest group is comprised of Sales Executives, totaling 326 employees. This highlights the significant emphasis placed on sales-related functions within the organization.

Research Scientist: Following closely, there are 292 employees serving as Research Scientists, underscoring the company’s substantial investment in research and development activities.

Laboratory Technician: With 259 employees, this role reflects the considerable technical support required for research and development operations.

The distribution of employees across these job roles illustrates a broad spectrum of functional areas within the company, from sales and research to technical support and management. The higher number of employees in roles such as Sales Executive and Research Scientist aligns with the company’s focus on sales performance and research innovation. The variation in job roles and the alignment with department sizes and educational backgrounds provide valuable insights into the organizational structure and resource allocation within IBM.

An analysis of the gender distribution within the IBM dataset reveals that approximately 60% of the employees, totaling 882 individuals, are male. In contrast, the remaining 40%, representing 588 individuals, are female. This indicates a gender imbalance within the company, with a higher proportion of male employees compared to female employees.

Upon analyzing the gender distribution in relation to attrition, it appears that the higher attrition rate among males is primarily a reflection of their overall higher representation in the company. While more males are leaving compared to females, this is proportional to their greater presence within the organization. The attrition rates for both genders follow a similar distribution pattern, suggesting that gender does not have a significant influence on attrition. The data indicates that attrition is relatively consistent across genders, with no evidence that gender plays a decisive role in employee turnover.

An examination of the graph reveals that males exhibit a higher frequency of business travel compared to females. Specifically, among those who travel very frequently, males have a higher count (160) compared to females (117). For those who travel infrequently, males also show a greater count (621) than females (422). Conversely, for individuals who do not engage in business travel, males have a count of 1,001, while females have a count of 49.

At first glance, these figures might suggest a notable relationship between gender and business travel frequency. However, it is crucial to consider the gender distribution within the company. Males constitute approximately 60% of the workforce, leading to a potential bias in the data. This bias could skew the apparent relationship between gender and business travel, reflecting more of the overall distribution rather than a significant underlying trend.

In examining the marital status of employees, we find that 45% (673 individuals) are married, 32% (470 individuals) are single, and 22% (327 individuals) are divorced. This distribution raises an interesting question about whether marital status has any influence on attrition. While marital status may not serve as a direct predictor of attrition, it could act as an indirect factor. Further analysis is required to determine if there is any significant relationship between marital status and employee turnover.

An interesting observation emerges when we examine the relationship between marital status and attrition. While the overall distribution shows that the majority of employees are married (the fewest are divorced), a closer look at the attrition data reveals a notable pattern. Among employees who left the company, the highest proportion were single (120 individuals), followed by married employees (84), and divorced employees (33).

However, a significant portion of employees remained with the company: 350 singles, 589 married, and 294 divorced employees. Given the notable class imbalance in the attrition data, it is challenging to draw definitive conclusions. Nevertheless, these figures suggest a potential pattern between marital status and employee attrition, warranting further investigation to better understand its implications.

Marital status and business travel seems to be uncorrelated.

An analysis of overtime patterns reveals that the majority of employees, around 71%, do not engage in overtime, while 29% do. When examining the relationship between overtime and attrition, no clear correlation emerges. Among those who do overtime, 127 employees left the company, while 110 remained. Given the closeness of these numbers, it is difficult to draw any definitive conclusions. Thus, the preliminary finding suggests that overtime does not appear to be significantly correlated with attrition at this stage of the analysis.

The relationship between overtime and business travel does not exhibit any notable correlation. A significant majority of individuals do not engage in overtime, and the overall distribution of business travel relative to overtime follows a similar pattern. The observed spike in business travel among individuals who do not work overtime can be attributed to the larger total number of non-overtime employees. There is no distinct pattern in the business travel distribution for different overtime categories. Individuals who travel infrequently constitute the largest group, followed by those who travel frequently and those who do not travel at all. This distribution is influenced by the inherent class imbalance within the business travel data.

## Numerical Features
Descriptive statistics are relevant primarily for continuous numerical features, as they provide meaningful insights into these data points. In contrast, categorical features require different types of analysis. Therefore, our initial focus will be on exploring and analyzing the continuous numerical features to understand their distributions and characteristics.a

This summary provides descriptive statistics for the Age feature in the dataset. Here’s a breakdown of each metric:

* Count: 1470 – The total number of entries for the Age feature, indicating that there are 1470 age values in the dataset.

* Mean: 36.92 – The average age of employees, showing that on average, employees are around 37 years old.

* Standard Deviation (std): 9.14 – This measures the dispersion or spread of the age values around the mean. A standard deviation of 9.14 indicates moderate variability in employee ages.

* Minimum (min): 18 – The youngest age in the dataset is 18 years old.

* 25th Percentile (25%): 30 – 25% of employees are 30 years old or younger.

* Median (50%): 36 – The middle value of the age distribution, with 50% of employees being 36 years old or younger and 50% being older.

* 75th Percentile (75%): 43 – 75% of employees are 43 years old or younger, indicating that the age distribution is skewed towards younger employees.

* Maximum (max): 60 – The oldest age in the dataset is 60 years old.

These statistics give a comprehensive view of the age distribution among employees, highlighting both the central tendency and the spread of age values.

The descriptive statistics for age reveal several insights about the workforce composition of the company or industry. The average age of employees is approximately 37 years, suggesting a preference for relatively younger individuals. This aligns with common industry practice, as companies often seek younger employees who can contribute significantly and adapt quickly. However, this does not necessarily mean that the workforce is exclusively young; it includes employees with valuable experience and up-to-date knowledge, typically ranging from their 20s to 30s.

The minimum age of 18 reflects standard legal requirements for employment, ensuring that employees are of legal working age. The maximum age extends to 60, which may indicate either the presence of older, experienced employees or potential outliers. The significant gap between the 75th percentile age of 43 and the maximum age suggests that while most employees are relatively young, there are some who are considerably older. This disparity highlights the variability within the age distribution, with a substantial portion of employees being concentrated in the younger demographic, but with a notable presence of older employees as well.

The distribution of the Age feature in the dataset provides valuable insights into the demographic characteristics of employees within the IBM tech industry. Key points derived from the analysis are as follows:

* Distribution Shape:
The histogram and violin plot for the Age feature suggest a distribution that approximates a normal distribution. This symmetry is typically considered advantageous for statistical analysis, indicating a balanced representation of age within the dataset.

* Range and Central Tendency:
The age of employees ranges from 18 to 60 years, with a noticeable concentration around the age of 35. The peak frequency at this age suggests that a large proportion of the workforce is in the mid-career phase, likely indicating individuals who possess significant experience and are actively contributing to their respective roles.

* Skewness and Symmetry:
The distribution demonstrates a slight rightward skew, as evidenced by the median being higher than the mean. However, this skewness is minimal, suggesting that the dataset remains largely symmetric. The slight skew indicates that there are relatively fewer older individuals, which slightly pulls the distribution's tail to the right.

* Age Concentration:
Most employees are found within the age bracket of 29 to 45 years, a range that encompasses professionals in their prime working years. This concentration highlights the typical career progression in the tech industry, where individuals are more likely to be established in their roles during these years.

* Decline in Older Age Groups:
Beyond the age of 45, there is a noticeable decline in the number of employees. This could be due to various factors, including retirements, career changes, or industry trends that favor younger professionals. The decline may reflect the dynamics of the tech industry, which often sees rapid innovation and a demand for emerging skill sets that may skew towards a younger demographic.

* Highly Experienced Individuals:
The small group of employees over the age of 45 likely represents individuals with considerable experience and expertise. These employees may occupy senior or specialized roles and potentially command higher compensation due to their advanced skill levels and experience. This assumption can be explored further through additional analysis of compensation data in relation to age.

* Minimum Age and Workforce Entry:
The minimum age of employees in the dataset is 18 years, corresponding with the legal working age in many countries. The presence of a small number of individuals at this age suggests that new entrants to the workforce are represented, albeit minimally, in the dataset.

* Real-World Dynamics:
The age distribution closely mirrors real-world employment patterns within the tech industry. The peak at mid-career ages, coupled with a gradual tapering off of older employees, reflects industry trends where younger professionals tend to dominate the workforce, while older, more experienced employees form a smaller, yet crucial, portion of the employee population.

This analysis of the Age feature provides a clear understanding of the age composition of employees within the tech industry and highlights key trends, such as the concentration of mid-career professionals and the presence of highly experienced senior employees. These findings open up avenues for further research, including the relationship between age, compensation, and career progression within the tech industry.

We have already seen that there exists a class imbalance in our data set so we will have to look at this particular observation without by keeping class imbalance in our mind obviously there are more individuals who are not leaving company and there are less individuals who are leaving the company and by looking at the violent plot we can see that there is a very slight difference in their overall statistics if we look at the statistics of those who do not leave the company then the median age turns out to be 36 whereas for those who do leave the company the median age turns out to be 32 so we can say that those who are leaving the company on an average might leave the company in their 30s now this is not a true fact and this might not be true but the median seems to be 32. Previously we saw that it on as a general perspective our data is following a normal distribution now it is still following a normal distribution when we look at the individuals who are not leaving the company but when it comes to the individuals who are leaving the company the overall distribution doesn't seem to be normal at all it seems to be some kind of initially uniform than it goes through an ups and downs and sees some peaks such as the peaks are observed at the age of 29 and age 31 where the count is 18 and then the P goes down and towards the higher ages there seems to be extremely low values so this do shows us that there are even individuals who can leave the company at almost any age majority of them might leave in their 20s and 30s and some will leave an extremely high ages after working now this might be time end or they do leave by their own wish although the problem is we cannot conclude anything about the overall distribution because the tourism values is quite low,

An examination of the age distribution by gender reveals that the distribution is nearly identical between males and females. The histogram analysis shows a balanced and uniform spread of ages across both genders, indicating no significant differences in the age composition between male and female employees within the dataset.

An analysis of business travel in relation to age reveals that there is no significant variation in the age distribution across different travel categories. Approximately 70% of the individuals travel infrequently, and the age distributions for all categories—those who travel frequently, rarely, or not at all—follow a normal distribution. The median age for those who travel rarely and for those who do not travel at all is 36, while the median age for frequent travelers is 35. The distribution of lower and upper age values is also quite similar across the groups. There are no distinct subgroups formed between age and business travel, with the highest concentration of individuals consistently falling within the age range of 29 to 38, or more broadly, in their 20s and 30s. This uniformity suggests no clear relationship between age and business travel frequency in this dataset.

