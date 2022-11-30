# Accurate Fairness Criterion
Code for Accurate Fairness: Improving Individual Fairness without Trading Accuracy (AAAI-23)
**************************************************************************************
Package Requirements:
- tensorflow 2.4.1
- keras 2.4.3 
**************************************************************************************
Siamese fairness in-processing for improving the accurate fairness of a machine learning model.

For example: to improve the accurate fairness on the Ctrip dataset,
- First run prepare_ctrip_data.py to generate the augmentated training dataset,
- Then run train_ctrip_siamese_fairness.py to improve the accurate fairness.
**************************************************************************************
How to check whether the predications of a machine learning model are accurately fair?

For example: to check the accurate fairness on the Ctrip dataset,
- First run prepare_ctrip_data.py to generate the test dataset,
- Then run get_ctrip_result.py to print out the accuracy, individual fairness, group fairness and accurate fairness measurements of the baseline model and the corresponding Siamese fairness model, in the following formats
  - a table, similar to Table 3 in our main paper, for statistical results of each model with various sensitive attributes
  - a chart X_Y.pdf in .\pic, similar to Figure 2 in our main paper, for fairness confusion matrix performances of dataset X under method Y (e.g., bl=baseline, sf=Siamese fairness) 
  - a chart X_Fairea.pdf in .\pic, similar to Figure 3 in our main paper, for Fairea evaluation of dataset X
