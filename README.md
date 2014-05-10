datacleaningcoursera
====================

Repository for the Coursera Course "Peer Assessments /Getting and Cleaning Data Project"

The R script run_analysis.R that does the following:

1. Merges the training and the test sets to create one data set.
2. Extracts only the measurements on the mean and standard deviation for each measurement. 
    *  Only considered features containing "mean()" and "std()", did not consider "meanFreq()" .
3. Uses descriptive activity names to name the activities in the data set
4. Appropriately labels the data set with descriptive activity names. 
5. Creates a second, independent tidy data set with the average of each variable for each activity and each subject. 
    *  This tidy dataset is saved into a text file called **"second.txt"**.
	*  Used name abbreviations "F_i" for the i-th feature, since the feature names are long, the names are described in the codebook
	
Used the following code for extracting mean and std measurements:
~~~R
features = read.table("features.txt")
features = features[grepl("mean\\(\\)", features[,2]) | grepl("std\\(\\)", features[,2]),]
~~~


