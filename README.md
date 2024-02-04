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
  + Less than 3% claims received InscClaimAmtReimbursed more than $375000
  + 1 attending physician, PHY422134 involved in more than 400 claims, while the average number of claims involved for other 11604 physicians is 3.45

+ **Train_ Outpatientdata.csv**: details about the claims filed for those patients who visit hospitals and are not admitted.
  + 133980 unique beneficiaries
  + 517737 unique claims submitted
  + 2 providers: PRV51459 and PRV53797 submitted more than 10355 and 5177 claims, while the average number of submitted cases for the other 5012 Providers is 100.7
  + Less than 0.1% of claims received InscClaimAmtReimbursed more than $4100
  + 1 attending physician, PHY422134 involved in more than 400 claims, while the average number of claims involved for other 11604 physicians is 3.45 

## Strategic Insights: Tackling Healthcare Fraud

## Exploratory Data Analysis

### Data Manipulation


### Feature Engineering


### Model Selection and Fitting



## Evaluating Success: Model Results & Implications




## Deployment & On-demand Prediction


## Limitations & Future Work
