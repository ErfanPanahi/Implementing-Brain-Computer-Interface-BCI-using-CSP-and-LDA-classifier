# Implementing-Brain-Computer-Interface-BCI-using-CSP-and-LDA-classifier
In this repository, I implement Brain Computer Interface (BCI) using CSP (Constraint Satisfaction Problem) and LDA (Linear Discriminant Analysis) classifier.

**Signal Recording Protocol:** For the purpose of conducting the second National Brain-Computer Interface (BCI) competition, signals from 15 healthy right-handed individuals (5 females and 10 males) with an average age of 31 years have been recorded. The signal acquisition system utilized 64 channels with a sampling frequency of 2400 Hz, and a power line filter was active during data recording to eliminate city power noise. The signal ground was obtained from the forehead, and one of the channels was connected to the right ear as a reference channel (which has been removed from the dataset, resulting in 63 available data channels).

All electrode signals have been low-pass filtered with a cut-off frequency of 50 Hz. The arrangement of electrodes is shown in the data matrix in the following figures, according to their names and row numbers.

![image](https://github.com/ErfanPanahi/Implementing-Brain-Computer-Interface-BCI-using-CSP-and-LDA-classifier/assets/107314081/f012dd60-7a82-4fb4-b4ef-466bac72e35f)

![image](https://github.com/ErfanPanahi/Implementing-Brain-Computer-Interface-BCI-using-CSP-and-LDA-classifier/assets/107314081/02222043-e173-4a23-82f4-d16631f23b85)

**Experimental Protocol:** Person 1 is comfortably seated on a chair, and in front of them, at a distance of half a meter, there is a display screen. The task that the person must perform during the experiment is executing a motion. The targeted body parts are the right wrist, right foot, and right arm. The experimental protocol is illustrated in the following figure.

![image](https://github.com/ErfanPanahi/Implementing-Brain-Computer-Interface-BCI-using-CSP-and-LDA-classifier/assets/107314081/d3eb52e9-c755-4a2f-9e0d-a334242d2461)

Initially, the "+" sign is displayed in the center of the screen for 2 seconds; during this time, the individual should not think about anything, and preparation for observing the signal should be in place. After 2 seconds, the desired signal appears; three types of signals have been used in this protocol, indicating the execution of the mentioned body movements to the individual. After a 2-second interval, the signal fades, and the word "Go" appears for 0.5 seconds. The individual will have 3 seconds from the moment of seeing "Go" to perform the intended action. In other words, depending on the type of signal, they should execute the corresponding movement.

**Competition Data Format:** Data for 15 individuals will be provided to the participants. For each individual, a file named subj_i.mat has been provided, where i represents the individual's number.

Dataset: [Link](https://drive.google.com/file/d/1h_Xi0ms4kpCvzSsMOsGfhhUqYezXk3s_/view?usp=sharing)

The 'data' is a cell with a size of 1×4, where each cell element contains EEG signals of an individual from the moment the word "Go" appears for a duration of 3 seconds. The first element of the cell contains EEG recorded during arm movement execution (class 1), the second element contains EEG recorded during finger flexion movement (class 2), the third element contains EEG recorded during leg movement execution (class 3), and the fourth element contains EEG recorded in the resting state without movement (class 4). The data for each element is stored in a three-dimensional array, where the first dimension represents the channel, the second dimension represents the sample points (4 in total), and the third dimension represents the number of trials corresponding to each movement class.

**Motion Data Separation**
In these data, there are 4 different motion states or classes, as follows:

- Class 1: Right Arm
- Class 2: Right Wrist Shake
- Class 3: Right Leg
- Class 4: No Movement

Based on various experiments and data observations, separating Class 4 from the other classes is easier. Additionally, Class 3 data exhibits greater differences compared to Classes 1 and 2.

In all three classifiers, each of which operates in a binary manner, we utilize pre-processing steps and the same classifiers. The only difference lies in the parameters considered for each classifier, which will be explained later. Additionally, the leave-one-out method is employed to evaluate the proposed classifiers.

***First Classifier:*** In this classifier, we decide whether the target test data belongs to class 4 or to the other 3 classes. Essentially, we are considering a binary classifier. For this classifier, depending on the specific subject, data from specific channels are selected. However, for certain subjects, data from all channels are taken into account. For example, for subject 1, the following channels are considered:

![image](https://github.com/ErfanPanahi/Implementing-Brain-Computer-Interface-BCI-using-CSP-and-LDA-classifier/assets/107314081/6175669c-d980-4307-b279-998e7860923f)

Next, we pass the data of the selected channels through a bandpass filter, the parameters of which vary based on the desired subject. For instance, for subject 1, the bandpass filter range is set as [20 40] Hz.

Then we use the Common Spatial Pattern (CSP) filters, which map the data from two classes to a space where one class has very high scatter and the other class has very low scatter. The CSP method reduces the data dimensions to m channels, which can vary depending on the subject, generally ranging from 3 to 5.

Finally, we extract a feature from each new channel, which is its variance. Then, we employ Linear Discriminant Analysis (LDA) as a classifier to separate the data, projecting them onto a line to maximize their separability. If the test data class is recognized as class 4, we stop here. However, if it belongs to 3 other classes, we move on to the second classifier.

***Second Classifier:*** In this classifier, we decide whether the test data belongs to class 3 or classes 1 and 2. In this classifier, data from all channels are used. Again, similar to the first classifier, we employ the same bandpass filtering, the CSP method, and the LDA classifier.

The main difference in this classifier compared to the first one is in the selection of the filtering interval, which also depends on the subject. For example, for subject 1, the bandpass filtering interval is set to [0.1 7] Hz.

Finally, if the test data class is recognized as class 2, we stop here. However, if it belongs to the other 2 classes, we proceed to the third classifier.

***Third Classifier:*** In this classifier, we decide whether the test data belongs to class 2 or class 1. The description of this classifier is identical to the second classifier, and the difference lies in the selection of the filtering interval, which also depends on the subject. For example, for subject 1, the bandpass filtering interval is set to [0.1 4] Hz.

At the end of this classifier, the class of the test data will be determined.

**Reslts for Subject 1:**

Confusion Matrix: (Using CSP (Constraint Satisfaction Problem) Algorithm and LDA (Linear Discriminant Analysis) classifier)

![image](https://github.com/ErfanPanahi/Implementing-Brain-Computer-Interface-BCI-using-CSP-and-LDA-classifier/assets/107314081/2c90090e-b403-4bdc-9e44-bd2c811a72d4)

