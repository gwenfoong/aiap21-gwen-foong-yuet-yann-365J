**AIAP 21 Technical Assessment**

Full Name: Gwen Foong Yuet Yann

Email Address: gwenfoong@hotmail.com



**Overview of the submitted folder and the folder structure**
├── eda.ipynb                 
├── src
│   ├── config.py             
│   ├── data_loader.py        
│   ├── preprocess.py         
│   ├── models.py             
│   └── train.py              
├── run.sh                    
├── requirements.txt          
├── .github/workflows/github-actions.yml  
└── data/
    └── gas_monitoring.db     
gas_monitoring.db is excluded from the repository using .gitignore 



**Instructions for executing the pipeline and modifying any parameters**
1. gas_monitoring.db to be stored locally data/gas_monitoring.db
2. Install Dependencies
    pip install -r requirements.txt
3. Run the pipeline locally
    bash run.sh



**Summary of key EDA findings and impact on pipeline**
| EDA Observation                                                       | Action Taken in Pipeline                                    |
| --------------------------------------------------------------------- | ----------------------------------------------------------- |
| Missing values in CO₂ ElectroChemical Sensor & MetalOxideSensor_Unit3 | Median imputation                                           |
| Missing values in CO Gas Sensor and Ambient Light Level               | Mode imputation                                             |
| Physically unrealistic sensor readings (Temperature, Humidity, CO₂)   | Clipped to realistic ranges                                 |
| Duplicate rows                                                        | Removed before preprocessing                                |
| Class imbalance in Activity Level                                     | Models evaluated using macro F1 in addition to accuracy |



**Describe how the features in the dataset are processed**
| Attribute                 | Processing Applied |
|---------------------------|---------------------|
| General                   | Removed duplicates |
| Time of Day               | (no cleaning required) |
| Temperature               | Clipped to 10–50 °C |
| Humidity                  | Clipped to 0–100% |
| CO2_InfraredSensor        | (no cleaning required) |
| CO2_ElectroChemicalSensor | Filled with median value |
| MetalOxideSensor_Unit1    | (no cleaning required) |
| MetalOxideSensor_Unit2    | (no cleaning required) |
| MetalOxideSensor_Unit3    | Filled with median value |
| MetalOxideSensor_Unit4    | (no cleaning required) |
| CO_GasSensor              | Filled with mode value |
| Session ID                | Dropped (identifier) |
| HVAC Operation Mode       | Converted to lowercase and trimmed spaces |
| Ambient Light Level       | Filled with mode value; added space between capital letters; replaced space with underscore; converted to lowercase |
| Activity Level (target)   | Converted to lowercase with underscores between words |



**Explanation of your choice of models for each machine learning task**

Logistic Regression >> Linear baseline that is simple and interpretable

Random Forest >> As there is multicollinearity after EDA, random forest handles multicollinearity and non-linear relationships well

Gradient Boosting >> Able to capture complex interactions between features 



**Evaluation of the models developed. Any metrics used in the evaluation should also be explained**
| Model               | Accuracy   | Macro F1   | Notes                                            |
| ------------------- | ---------- | ---------- | ------------------------------------------------ |
| Logistic Regression | 0.5025     | 0.4536     | Struggles with nonlinearities                    |
| **Random Forest**   | **0.6822** | **0.5178** | **Best overall — balanced across classes**       |
| Gradient Boosting   | 0.6594     | 0.4551     | Good on moderate class but weak on high activity |



**Other considerations for deploying the models developed**

To include valid input ranges to handle sensor's capturing of impossible values. 
