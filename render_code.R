

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


