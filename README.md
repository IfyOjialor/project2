# Bike Share Analysis

As part of the second project for the ST 558(Data Science) Course, I have come up with an analysis on the bike share project.

Over the past decade, bicycle-sharing systems have been growing in number and popularity in cities across the world. Bicycle-sharing systems allow users to rent bicycles for short trips, typically 30 minutes or less. The wealth of data from these systems can be used to explore how these bike-sharing systems are used.

In this project, I have performed an exploratory data analysis on the data provided by the UCL Machine Learning repository available [here] (https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). We used to methods to model the cnt variable: a regression tree and boosted tree model. The weekday variable was automated to include files for each weekday. The sub-documents are avaiable below;

* The analysis for [Monday is available here]("blob/main/1.md")

* The analysis for [Tuesday is available here]("blob/main/2.md")

* The analysis for [Wednesday is available here]("blob/main/3.md")

* The analysis for [Thursday is available here]("blob/main/4.md")

* The analysis for [Friday is available here]("blob/main/5.md")

* The analysis for [Saturday is available here]("blob/main/6.md")

* The analysis for [Sunday is available here]("blob/main/0.md")

Libraries required for this analysis;
library(readr)
library(ggplot2)
library(corrplot)
library(caret)
library(rmarkdown)
library(dplyr)

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
