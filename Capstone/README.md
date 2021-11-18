# **Introduction**

## **Invasive Ductal Carcinoma**
Breast cancer is the most frequent form of cancer affecting women  [[source]](https://www.nccs.com.sg/patient-care/cancer-types/cancer-statistics). It accounts for 29.8% of all cancer diagnoses between the period of 2014 - 2018.  

Of all breast cancer subtypes, Invasive Ductal Carcinoma (IDC) is the most common subtype, representing 80 percent of all breast cancer diagnoses. [[source]](https://www.hopkinsmedicine.org/breast_center/breast_cancers_other_conditions/invasive_ductal_carcinoma.html).  IDC is known to grow in the milk duct and invade the fibrous or fatty tissue of the breast outside of the duct. [[source]](https://www.hopkinsmedicine.org/breast_center/breast_cancers_other_conditions/invasive_ductal_carcinoma.html)

## **Breast Cancer Histopathology**

One way to identify IDC is through a histopathological examination, where a tissue is extracted from a patient and then studied under a microscope by a pathologist.  The pathologist then examines the spread of IDC and assigns an aggressiveness grade to the sample (grade of cancer). (i.e. the more aggressive the spread of cancer, the higher a grade/stage is given) [[source]](https://www.nhs.uk/common-health-questions/operations-tests-and-procedures/what-do-cancer-stages-and-grades-mean/)

The analysis of breast cancer biopsys is time-consuming and at times, inaccurate. Pathologists have to go through swathes of benign regions in tissue samples to identify areas with IDC and the extent of its spread.
In addition, the variability of appearance in hematoxylin and eosin (H&E) stained areas presents one of the major barriers to accurate image analysis. Due these factor, detecting and categorizing these deviations can be difficult even for experienced pathologists [[source]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7465368/).

As such, a machine learning approach is believed to increase efficiency and accuracy, allowing the pathologist to carry out a more detailed appraisal on histopathology scans.  This machine learning algorithm should not replace the pathologist's examination completely but serve as a very useful tool in the pathologist's arsenal. 


# **Problem Statement**

A research firm would like to use deep learning to diagnose the presence of IDC in breast histopathological images.  This is an image binary classification problem and the objective is to be able to accurately classify image patches into 2 groups: presence of IDC or absence of IDC.  In particular, the firm is interested in obtaining a reasonably high balanced accuracy score and recall score.  Elaboration on the chosen metrics will be provided later.


# **Dataset**

The dataset was obtained from this [link](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) on kaggle.  

The dataset consist of extractions from Whole Slide Images. Each Whole Slide Image is split into non-overlapping coloured image patches that are 50 by 50 pixels.  Irrelevant patches (mostly fatty tissue or slide background) were discarded. Regions containing IDC were manually annotated by experienced pathologists.


# **Model Evaluation Metrics**


The classification model chosen had a twofold objective to 1) reduce the number of false negatives (optimal recall score) and 2) attain a higher Balanced Accuracy Score.  

Balanced Accuracy Score is a more meaningful evaluation metrics compared to accuracy score, especially in the case of an imbalanced dataset.  It is the average of the proportion correctly classified of each class.

In the case of breast cancer, it is imperative that we minimise false negative as incorrectly predicting ‘no IDC presence’ when there is IDC will have serious consequences, hence recall score is something we wanted to focus on.


# **Model Workflow and Evaluation**

After pre-processing the data, we create a simple baseline model.
![alt text](https://github.com/eg7777/projects/blob/main/Capstone/image/Model%20Workflow.png?raw=true)

Less complex models like the custom model performed better than more complex models such as Resnet50 and VGG16.

Although the baseline and ResNet50 model achieved a specificity score of 0.97, the corresponding balanced accuracy and recall scores were low and thus, neither models were selected as a chosen model.

The chosen custom model had the highest balanced accuracy (0.815) and recall (0.761) score.  

|Model|Balanced Accuracy|Specificity|Recall|
|------------------------|-------|-------|-------|
|Baseline Model|0.73251|0.97063|0.494389|
|Vgg16 model without image augment|0.80283|0.945429|0.660231|
|Vgg16 model with image augmentation|0.792329|0.946869|0.637789|
|ResNet50 model without image augmentation|0.699551|0.97171|0.427393|
|ResNet50 model with image augmentation|0.734626|0.948459|0.520792|
|Model without image augmentation|0.805593|0.889239|0.721947|
|Model with image augmentation|0.815037|0.869019|0.761056|

The custom model was further tuned. As can be seen below, the custom model with image augmentation v3 provided us a reasonably higher recall and balanced accuracy score than the other custom models.  At the same time, it was chosen as it had a fewer number of false positives than model v2, v6, v7.  

|Model Versions|Image Augmentation|Dropout Layer Value|Hidden Layer Node|Batch Normalization|Balanced Accuracy|Specificity|Recall|Remarks|
|--------|--------|--------|--------|--------|--------|--------|--------|-----------------|
|v0|No|[0.25,0.25]|[128,64,2]|No|0.805593|0.889239|0.721947||
|v1|Yes|[0.25,0.25]|[128,64,2]|No|0.812226|0.843098|0.781353||
|v2|Yes|[0.5,0.5]|[128,64,2]|No|0.727017|0.513935|0.940099|Not chosen, larger number of false positives.|
|v3|Yes|[0.25,0.5]|[128,64,2]|No|0.815037|0.869019|0.761056|Chosen model|
|v4|Yes|[0.25,0.5]|[128,64,2]|Yes (after conv 2)|0.5347|0.97138|0.0980198||
|v5|Yes|[0.25,0.5]|[64,2]|No|0.806718|0.882579|0.730858||
|v6|Yes|[0.25,0.5]|[32,16,2]|No|0.814708|0.851859|0.777558|Not chosen, larger number of false positives.|
|v7|Yes|[0.25,0.5]|[64,32,2]|No|0.818196|0.843158|0.793234|Not chosen, larger number of false positives|
|v8|Yes|[0.25,0.5]|[128,64,32,2]|No|0.802418|0.777278|0.827558||
|v9|Yes|[0.25,0.5]|[128,64,2]|Yes (before pooling)|0.79222|0.880479|0.70396||


# **Conclusion**

The custom model is effective in meeting the objectives of the firm in predicting the presence/absence of IDC with a 81.5% balanced accuracy score and a 76.0% recall score.

**Save resources (time)**

It takes approximately 10 - 20 minutes for a pathologist to look through a histopathological image under a microscope. [Source] It took the model less than 1 minute to predict for the presence or absence of IDC for more than 39,000 image patches. Assuming each patient has 1,000 image patches, the model is able to predict the presence of IDC for approximately 39 people.  Coupled with their expertise, pathologist could use this deep learning method to better detect IDC in a shorter time frame.

**Less complex models had better detection ability**

Less complex models (fewer convolutional layers) appear to provide better detection ability.

## **Cost Benefit Analysis**

A cost benefit analysis is conducted to evaluate the implications of the machine learning algorithm, especially in terms of its potential in resource saving.   

As some of the data on breast cancer in Singapore cannot be readily found, both local and international findings have been referenced when making assumptions. 

**Breast Pathologist Salary**

Salary of a breast pathologist have been referenced from this [[link]](https://www.ziprecruiter.com/Salaries/Breast-Pathologist-Salary). Assuming a pathologist works 8 hours day, 5 days a week, 20 days per month, a rough estimation of how much they will earn is detailed below:

|Breast Pathologist Salary||
|------------------------------------------------------------|---------|
|Breast pathologist in Singapore makes per year|107291|
|Amount a breast pathologist earn per hour|55.88|


**Resources required for pathologists to evaluate histopathological images per year**

About 20% of all biopsies result in a diagnosis of breast cancer [[source]](https://effectivehealthcare.ahrq.gov/products/breast-biopsy-update/clinician)

**Cost without machine learning**

Based on our confusion matrix and the overall breakdown of IDC positive and negative images in the dataset, we can roughly assume that around 30% of all images will be identified positive for IDC-positive individuals. Hence, for this cost benefit analysis, we will assume the pathologist will manually review all regions in which the machine has identified to be IDC-positive. 

|Time and cost taken to detect for the presence of IDC (without machine learning)||
|------------------------------------------------------------|---------|
|Number of breast cancer diagnoses in Singapore from 2014 - 2018|11,232 cases|
|Estimated number of breast cancer diagnoses in Singapore per year|2808 cases|
|Estimated number of individual biopsy pathologists conduct per year, including those with non-cancer diagnoses|14,040 cases|
|Estimated time taken for a pathologist to review a histopathological image, without machine learning algorithm|0.25 hrs|
|Estimated time taken for a pathologist to review all images per year, without machine learning algorithm|3,510 hrs|
|Estimated cost (pathologist salary) to review all images per year, without machine learning|$196,138.8|


**Cost with machine learning**

|Time and cost taken to detect for the presence of IDC (with machine learning)||
|------------------------------------------------------------|---------|
|Approximate time taken for machine to detect presence of IDC for an individual|0.5 min (instantly to < 1 min)|
|Additional time taken for a pathologist to validate/review the machine's results (identify False Positive): 4.5 mins|4.5 min|
|Estimated time taken for a pathologist to review all images per year, with machine learning|1,053 hrs|
|Estimated cost (pathologist salary) to review all images per year, with machine learning|$58,841.64|


**Annual healthcare savings**

|Annual healthcare savings||
|------------------------------------------------------------|---------|
|Cost savings per year|$137,297.16|
|Healthcare professionals' hours saved per year|2,457 hrs|


## **Limitation**

**Dataset limitations**

**Fragmented image patches**

The dataset comprises of extracted (50 by 50 pixels) images of whole slide images.  The machine, hence, is trained on the features found on indivdual image patches, rather than from the entire slide.  It could potentially miss out on learning features as a whole and features that are connected between image patches that are cut off from the extracted dataset.


## **Next Step**

**Trained on individuals with no cancer**

One of the issues is that the dataset contains data from cancer patients only. Hence, the machine is trained on both the "normal" and "IDC-positive" image patches from cancer patients. It would be beneficial if it were also trained on individuals who did not have breast cancer.    

**Other methodologies**

Recently, there have been more research done in the field of medical imagery diagnosis using deep learning techniques. Other possible methodologies that can be explored include ensemble neural network models. Ensemble learning combines the predictions from multiple neural network models to reduce the variance of predictions and reduce generalization error. [[Source]](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/).  Additionally, image segmentation can be explored as a classification/prediction tool. Image segmentation aims to obtain region of interest. The parts into which the image is divided are based on the image properties (similarity, discontinuity etc). It aims to segment an image into different areas for better analysis. [[Source]](https://www.analyticsvidhya.com/blog/2021/09/image-segmentation-algorithms-with-implementation-in-python/)

**Whole Slide Imagery**

Explore utilising the whole slide imagery as opposed to relevant fragments.

**Aggressiveness Rating**

Explore automatically assigning aggressivness rating based on the extent of IDC-presence in whole slide imagery.
