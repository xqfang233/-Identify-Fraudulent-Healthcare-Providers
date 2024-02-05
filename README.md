# Identify Fraudulent Healthcare Providers
This project aims to detect fraudulent healthcare providers with machine learning algorithms, uncovering the nuanced patterns of healthcare data to distinguish between legitimate and fraudulent practices.

## Motivation
Healthcare fraud, the deceptive practice of trying to obtain unauthorized benefits or payments from healthcare systems, poses a significant challenge in the United States and costs billions annually. This malpractice can take many forms, including billing for services not rendered, upcoding for more expensive treatments, using another person's identity to receive healthcare services, or performing unnecessary procedures to increase revenue. 

Various entities involved in healthcare processes can perpetrate healthcare fraud, including but not limited to patients, providers, and payers. However, provider fraudulent activities are particularly concerning as they not only lead to financial losses but also compromise patient care and trust in the medical system. Addressing this issue is crucial for maintaining the efficacy and ethical standards of healthcare delivery. Therefore, this project is dedicated to detecting the fraudulent behaviors of Providers. 

## Data Overview
This [dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis), sourced from Kaggle, offers a comprehensive view into the dataset's distribution, including notable data abnormalities and outliers. Instead of revisiting these basics, we'll dive deeper into the significance of specific features and explore how they can be effectively utilized in our analysis and predictive modeling.

The dataset is from [kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) and the basic data distribution for each column can be checked on the same website, so I'm not going to rebuild the wheel here but rather discuss more about what some of the features mean and how to use them to form our analysis and prediction.

For this project, we will be using the following files:
+ **Train.csv**: only includes 2 columns
  + Provider: 5410 unique Provider ID
  + Potential fraud: "Yes" or "No"
+ **Train_Beneficiarydata.csv**: contains beneficiary demographic details
  + 138556 unique beneficiaries
  + 1% of beneficiaries (1239) were deceased
  + 14% of beneficiaries have Renal disease (RenalDiseaseIndicator == 'Y')
  + Some states/county has much higher frequency
  + Very few beneficiaries (<1% don't have 12 months of coverage Part A or Part B)

+ **Train_ Inpatientdata.csv**: details about the claims filed for those patients who are admitted to the hospitals
  + 31289 unique beneficiaries
  + 40474 unique claims submitted
  + 2 providers: PRV52019 and PRV55462 each submitted more than 400 claims, while the average number of submitted cases for the other 39572 Providers is 18.9
  + Less than 3% of claims received InscClaimAmtReimbursed more than $375000
  + 1 attending physician, PHY422134 involved in more than 400 claims, while the average number of claims involved for other 11604 physicians is 3.45

+ **Train_ Outpatientdata.csv**: details about the claims filed for those patients who visit hospitals and are not admitted.
  + 133980 unique beneficiaries
  + 517737 unique claims submitted
  + 2 providers: PRV51459 and PRV53797 submitted more than 10355 and 5177 claims, while the average number of submitted cases for the other 5012 Providers is 100.7
  + Less than 0.1% of claims received InscClaimAmtReimbursed more than $4100
  + 1 attending physician, PHY422134 involved in more than 400 claims, while the average number of claims involved for other 11604 physicians is 3.45 

## Strategic Insights: Tackling Healthcare Fraud
As this is a fraud detection problem, we should concentrate on data abnormalities, in other words, “outliers” in data. Based on the data we have, here are several abnormalities worth noticing:

1. Billing abnormalities
  +  Repeatedly billing for the highest-cost services (upcoding)
  +  Billing for services more frequently than usual
  +  Billing for services that are inconsistent with the patient’s diagnosis or medical history

2. Provider abnormalities:
    + Consistently bill at higher rates or frequency compared to their peers
    + High volume of services claimed for a small number of patients

3. Service Type and Frequency:
    + Unnecessary or excessive service for the diagnosis
    + Frequently billing for rare, complex, or expensive procedures

## Exploratory Data Analysis

The Data tables merging procedure:

![table_merging drawio](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/9f58e50b-79f8-4011-ab62-8e6f3a5124dc)


Based on the abnormalities discussed above, we created the following features for subsequent prediction. The detailed data analysis process is in *analytics and prediction.ipynb*.
1. Ot_BeneID: the number of unique out-patient beneficiaries this provider submitted claims for 
2. Ot_ClaimID: the number of unique claims this provider submitted for out-patients
3. Ot_InscClaimAmtReimbursed: the average amount the insurance company has paid back to either the healthcare provider or the patient for the medical service
4. Ot_DeductibleAmtPaid: the average amount that has been paid by the out-patients towards the deductible for services
5. Ot_claim_duration: the average duration from a claim starts until the claim ends (in days) for out-patients
6. Ot_ClmDiagnosisNum: the average number of diagnoses code for out-patients
7. Ot_physicianNum: the average number of physicians involved for a single claim for out-patients
8. Ot_ClmProcedureNum: the average number of procedures for a single claim for out-patients
9. Ot_RenalDiseaseIndicator: the total number of out-patients that has Renal diseases
10. Ot_IPAnnualReimbursementAmt: the average amount reimbursed annually for out-patient services that are associated with or arise from inpatient care
11. Ot_IPAnnualDeductibleAmt: the average amount that has been paid by the out-patients towards the deductible for services that are associated with or arise from inpatient care
12. Ot_OPAnnualReimbursementAmt: the average amount reimbursed annually for out-patient services that are associated with or arise from outpatient care
13. Ot_OPAnnualDeductibleAmt: the average amount that has been paid by the out-patients towards the deductible for services that are associated with or arise from outpatient care
14. Ot_location: the number of unique locations for out-patients
15. Ot_ChronicNum: the average number of chronic conditions each out-patient carries
16. Ot_passed away: the total number of deceased out-patients
17. Ot_12Months_PartACov: the ratio of out-patients that are under the "Part A" segment of a health insurance plan coverage for 12 months
18. Ot_12Months_PartBCov: similar to 17
19. In_BeneID: similar to 1, but for in-patients
20. In_ClaimID: similar to 2, but for in-patients
21. In_InscClaimAmtReimbursed
22. In_DeductibleAmtPaid
23. In_claim_duration
24. In_ClmDiagnosisNum
25. In_physicianNum
26. In_ClmProcedureNum
27. In_RenalDiseaseIndicator
28. In_IPAnnualReimbursementAmt
29. In_IPAnnualDeductibleAmt
30. In_OPAnnualReimbursementAmt
31. In_OPAnnualDeductibleAmt
32. In_location
33. In_ChronicNum
34. In_passed away
35. In_12Months_PartACov
36. In_12Months_PartBCov

![final_features1](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/614047f4-5080-47af-b262-ee3d360ab2db)
* example data for the final features * 


## Model Selection and Fitting
Here, I compared the performance for 5 models: Logistic Regression, Random Forest, XGBoost, lightGBM and KNN. Let's go through them one by one

### Logistic Regression
Logistic Regression is a straightforward statistical model that estimates probabilities using a logistic function, making it ideal for binary classification tasks like fraud detection. Its simplicity and interpretability make it a good baseline model, allowing for an easy understanding of how different features influence the likelihood of fraudulent activity.


![logistic-regression](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/45eca568-ff52-46d5-a8e6-1b2415681618)

Accuracy: 0.8664007976071785

### Random Forest
Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions. It is known for its high accuracy, robustness to overfitting, and ability to handle complex interactions between features, making it highly effective for detecting nuanced patterns of fraud in healthcare data.

![random-forest](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/a2fea866-0b15-46b9-b493-4d366f50d250)

Accuracy: 0.9322033898305084

### XGBoost
XGBoost is an implementation of gradient-boosted decision trees designed for speed and performance. It stands out for its efficiency and scalability, often delivering superior results even with default parameters. XGBoost's ability to handle sparse data and its feature importance scores make it particularly useful for identifying critical predictors of fraudulent behavior.

![xgb](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/33db1f2a-9546-4788-a5e2-234c3a71abc3)

Accuracy: 0.942173479561316


### KNN
KNN is a simple, non-parametric algorithm that classifies each data point based on the majority labels of its k nearest neighbors. Its simplicity and the intuitive nature of its decision-making process can be advantageous in fraud detection, particularly when the fraudulent patterns are closely clustered together in the feature space.

![knn](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/0e49a470-d8a9-40d4-9d4d-63683ff9f353)

Accuracy: 0.843469591226321

### lightGBM
LightGBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithms, optimized for speed and efficiency. Its innovative handling of categorical features and large datasets, along with minimal memory usage, makes LightGBM well-suited for fraud detection, especially when dealing with vast amounts of healthcare claims data.

![lgb](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/06526ad5-8af7-4b95-b1bd-52d6f3b53903)

Accuracy: 0.93419740777667


### Summary
+ The three tree-based model, XGBoost, lightGBM and random forest all achieved high accuracy
+ KNN had the worst accuracy performance by having the highest number of false-positives

## Evaluating Success: Model Results & Implications
Since our data is highly imbalanced (legitimate:fraudulent = 9:1), accuracy may not be the most proper metric to measure the model predicting performance. Therefore, besides of accuracy, we also apply ROC-AUC curve and Precision-Recall curve as performance measurements.

### ROC-AUC curve
The ROC (Receiver Operating Characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The AUC (Area Under the Curve) represents the measure of the ability of the classifier to distinguish between classes. 

![roc](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/e93799c8-caaa-4462-99cf-f3f3d76e566e)

### Precision-Recall curve
The precision-recall curve shows the trade-off between precision (the ratio of true positive predictions to all positive predictions) and recall (the ratio of true positive predictions to all actual positives) for different threshold values.

![precision-recall](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/dac08a79-4f9b-4695-834f-ed8232a9e8d3)


### Summary
Due to the imbalanced nature of our dataset, Precision-Recall and ROC-AUC curves are both critical measurements of model performance. Particularly, the Precision-recall curve is valuable because it focuses on the performance of the minority class (fraudulent transactions), which is usually of greater interest in fraud detection scenarios. ROC-AUC is also useful as it considers both the false positive rate and the true positive rate, providing a holistic view of model performance across different thresholds. Therefore, the **XGBoost** model had the best performance.


## Deployment & On-demand Prediction

### Predictive contributions for each feature
Since we have 36 features it's bothersome to enter them all at once for on-demand point prediction, we conducted a SHAP analysis for the XGBoost model to see which features are more important toward making predictions.

![shap-xgboost](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/bbeb2ca8-2fe1-4a63-9d88-16af4a4ace24)

The features are ordered descendingly based on their predictive contributions, and we pick the first 8 features to retrain an XGBoost model for on-demand analysis:

![xgb_t](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/3a493773-4443-4f6f-ad8a-79108fdda083)

Accuracy: 0.9212362911266201

### Deployment and Prediction
The simplified XGBoost model was deployed [here](https://dark-passage-413222.uc.r.appspot.com) on the Google Cloud platform. The code adapted for deployment can be found under the "gcp-deploy" branch.

When accessing the URL, we should be able to see the home page waiting for inputs of 8 features.

![local_deploy1](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/2824b357-838f-4df7-90a9-f0a5a6b38f2d)

After entering all 8 feature values, click the blue "Fraudulent or Not?" button to submit the values, and then you will see whether our model thinks how likely this provider is fraudulent.

![local_deploy2](https://github.com/xqfang233/Identify-Fraudulent-Healthcare-Providers/assets/81652429/c5af9fb7-89d7-45bb-b3bd-b2a903fbc794)


## Limitations & Future Work
### Limitations
Our model currently processes instances one at a time, limiting real-time analysis scalability. Additionally, the dataset's diversity and integrity are constrained by numerous missing values, impacting model robustness. The logistic regression model, in particular, wasn't scaled, potentially affecting its performance.

### Future Enhancements
+ Focusing on adapting the model to batch processing for improved efficiency
+ Addressing data quality through advanced imputation techniques could enrich dataset reliability
+ Implementing feature scaling and normalization practices for logistic regression


## references
1. [Kaggle datasets: HEALTHCARE PROVIDER FRAUD DETECTION ANALYSIS](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
2. Busch, Rebecca S. Healthcare fraud: auditing and detection guide. John Wiley & Sons, 2012.

