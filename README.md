# Bike Share Analysis

---
Author: Ifeoma Ojialor
Date: 10/16/2020

As part of the second project for the ST 558(Data Science) Course, I have come up with an analysis on the bike share project.

Over the past decade, bicycle-sharing systems has been growing in number and popularity in cities across the world. Bicycle-sharing systems allow users to rent bicycles for short trips, typically 30 minutes or less. The wealth of data from these systems can be used to explore how these bike-sharing systems are used.

In this project, I have performed an exploratory data analysis on the data provided by the UCL Machine Learning repository available [here] (https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). The purpose of this repo is to store the relevant files and information needed for predicting the total count of bikes in a bike share program. We used the regression tree and boosted tree model to predict the total count of bikes. As usual, EDA was conducted before building the model. The weekday variable was automated to include files for each weekday. The sub-documents are available below;

* The analysis for [Monday is available here](Monday.md)

* The analysis for [Tuesday is available here](Tuesday.md)

* The analysis for [Wednesday is available here](Wednesday.md)

* The analysis for [Thursday is available here](Thursday.md)

* The analysis for [Friday is available here](Friday.md)

* The analysis for [Saturday is available here](Saturday.md)

* The analysis for [Sunday is available here](Sunday.md)

The following libraries are required for this analysis;

library(readr)  
library(ggplot2)  
library(corrplot)  
library(caret)  
library(rmarkdown)  
library(dplyr)  
library(tidyverse)

Code used to automate the weekday parameter;
```
weekdays <- unique(day1$weekday)

#create filenames
output_file <- paste0(weekdays, ".md")

#create a list for each weekday with just the day parameter
params <- lapply(weekdays, FUN = function(x){list(days = x)})

#put into a data frame 
reports <- tibble(output_file, params)

reports 

library(rmarkdown)
apply(reports, MARGIN = 1, 
      FUN = function(x){
        render(input = "Proj2.Rmd", output_file = x[[1]], params = x[[2]])})

```
