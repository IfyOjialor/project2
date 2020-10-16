Project 2
================
Ifeoma Ojialor
10/16/2020

## Introduction

In this project, we will use a bike-sharing dataset to create machine
learning models. Before moving forward, I will briefly explain the
bike-sharing system and how it works. A bike-sharing system is a service
in which users can rent/use bicycles on a short term basis for a fee.
The goal of these programs is to provide affordable access to bicycles
for short distance trips as opposed to walking or taking public
transportation. Imagine how many people use these systems on a given
day, the numbers can vary greatly based on some elements. The goal of
this project is to build a predictive model to find out the number of
people that use these bikes in a given time period using available
information about that time/day. This in turn, can help businesses that
oversee this systems to manage them in a cost efficient manner.  
We will be using the bike-sharing dataset from the UCL Machine Learning
Repository. We will use the regression and boosted tree method to model
the response variable `cnt`.

## Exploratory Data Analysis

First we will read in the data using a relative path.

``` r
#read in data and filter to desired weekday
day1 <- read.csv("Bike-Sharing-Dataset/day.csv")
head(day1,5)
```

    ##   instant     dteday season yr mnth holiday
    ## 1       1 2011-01-01      1  0    1       0
    ## 2       2 2011-01-02      1  0    1       0
    ## 3       3 2011-01-03      1  0    1       0
    ## 4       4 2011-01-04      1  0    1       0
    ## 5       5 2011-01-05      1  0    1       0
    ##   weekday workingday weathersit     temp
    ## 1       6          0          2 0.344167
    ## 2       0          0          2 0.363478
    ## 3       1          1          1 0.196364
    ## 4       2          1          1 0.200000
    ## 5       3          1          1 0.226957
    ##      atemp      hum windspeed casual registered
    ## 1 0.363625 0.805833  0.160446    331        654
    ## 2 0.353739 0.696087  0.248539    131        670
    ## 3 0.189405 0.437273  0.248309    120       1229
    ## 4 0.212122 0.590435  0.160296    108       1454
    ## 5 0.229270 0.436957  0.186900     82       1518
    ##    cnt
    ## 1  985
    ## 2  801
    ## 3 1349
    ## 4 1562
    ## 5 1600

Next, we will remove the *casual* and *registered* variables since the
`cnt` variable is a combination of both.

``` r
day1 <- select(day1, -casual, -registered) 
day1$weekday <- as.factor(day1$weekday)
levels(day1$weekday) <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
day <- filter(day1, weekday == params$days)

#Check for missing values
miss <- data.frame(apply(day,2,function(x){sum(is.na(x))}))
names(miss)[1] <- "missing"
miss
```

    ##            missing
    ## instant          0
    ## dteday           0
    ## season           0
    ## yr               0
    ## mnth             0
    ## holiday          0
    ## weekday          0
    ## workingday       0
    ## weathersit       0
    ## temp             0
    ## atemp            0
    ## hum              0
    ## windspeed        0
    ## cnt              0

There are no missing values in the dataset, so we can continue with our
analysis.

``` r
#Change the variables into their appropriate format.
day$season <- as.factor(day$season)
day$weathersit <- as.factor(day$weathersit)
day$holiday <- as.factor(day$holiday)
day$workingday <- as.factor(day$workingday)
day$weekday <- as.factor(day$weekday)
day$yr <- as.factor(day$yr)
day$mnth <- as.factor(day$mnth)

levels(day$season) <- c("winter", "spring", "summer", "fall")
levels(day$yr) <- c("2011", "2012")
str(day)
```

    ## 'data.frame':    105 obs. of  14 variables:
    ##  $ instant   : int  3 10 17 24 31 38 45 52 59 66 ...
    ##  $ dteday    : chr  "2011-01-03" "2011-01-10" "2011-01-17" "2011-01-24" ...
    ##  $ season    : Factor w/ 4 levels "winter","spring",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ yr        : Factor w/ 2 levels "2011","2012": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ mnth      : Factor w/ 12 levels "1","2","3","4",..: 1 1 1 1 1 2 2 2 2 3 ...
    ##  $ holiday   : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 2 1 1 ...
    ##  $ weekday   : Factor w/ 7 levels "Sunday","Monday",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ workingday: Factor w/ 2 levels "0","1": 2 2 1 2 2 2 2 1 2 2 ...
    ##  $ weathersit: Factor w/ 3 levels "1","2","3": 1 1 2 1 2 1 1 2 2 1 ...
    ##  $ temp      : num  0.1964 0.1508 0.1758 0.0974 0.1808 ...
    ##  $ atemp     : num  0.189 0.151 0.177 0.118 0.186 ...
    ##  $ hum       : num  0.437 0.483 0.537 0.492 0.604 ...
    ##  $ windspeed : num  0.248 0.223 0.194 0.158 0.187 ...
    ##  $ cnt       : int  1349 1321 1000 1416 1501 1712 1913 1107 1446 1872 ...

### Univariate Analysis

The `cnt` is the response variable, so we’ll use a histogram to get a
visual understanding of the variable.

``` r
ggplot(day, aes(x = cnt)) + theme_bw() + geom_histogram(aes(y =..density..), color = "black", fill = "white", binwidth = 1000) + geom_density(alpha = 0.2, fill = "blue") + labs(title = "Count Density", x = "Count", y = "Density")
```

![](Monday_files/figure-gfm/cnt-1.png)<!-- -->

``` r
summary(day$cnt)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      22    3310    4359    4338    5875    7525

From the histogram and summary statistics output, it is pretty evident
that the count of total rental bikes are in the sub 5000 range. We will
investigate if there is a relationship between the response variable and
other relevant predictor variables in the next section. Lets look at the
other variables individually.

``` r
#visualize numeric predictor variables using a histogram
p1 <- ggplot(day) + geom_histogram(aes(x = temp), fill = "red", binwidth = 0.03)
p2 <- ggplot(day) + geom_histogram(aes(x = atemp), fill = "red", binwidth = 0.03)
p3 <- ggplot(day) + geom_histogram(aes(x = hum), fill = "red", binwidth = 0.025)
p4 <- ggplot(day) + geom_histogram(aes(x = windspeed), fill = "red", binwidth = 0.03)
gridExtra::grid.arrange(p1,p2,p3,p4, nrow = 2)
```

![](Monday_files/figure-gfm/EDA_num-1.png)<!-- -->

Observations:  
\* No clear cut pattern in `temp`and `atemp`.

  - `hum` appears to be skewed to the left when the dataset is not
    filtered to a specific weekday.

  - `windspeed` appears to be skewed(right). This variable should be
    transformed to curb its skewness.

  - The distribution of `temp` and `atemp` looks very similar. We should
    think about taking out one of the variables.

<!-- end list -->

``` r
#visualize categorical predictor variables
h1 <- ggplot(day) + geom_bar(aes(x = season),fill = "pink")
h2 <- ggplot(day) + geom_bar(aes(x = yr),fill = "pink")
h3 <- ggplot(day) + geom_bar(aes(x = holiday),fill = "pink")
h4 <- ggplot(day) + geom_bar(aes(x = workingday),fill = "pink")
h5 <- ggplot(day) + geom_bar(aes(x = mnth),fill = "pink")
h6 <- ggplot(day) + geom_bar(aes(x = weathersit),fill = "pink")
gridExtra::grid.arrange(h1,h2,h3,h4,h5,h6, nrow = 3)
```

![](Monday_files/figure-gfm/EDA_cat-1.png)<!-- -->

Observations:  
\* The variation between the four seasons is little to none.

  - About the same number of people rode bikes in 2011 and 2012.

  - Many people rode bikes on days that are not holidays.

  - Most people used the bike-sharing system on days that were neither
    weekends nor holidays.

  - Most people used the bike sharing system on days with clear weather.

### Bi-variate Analysis

In this section, we will explore the predictor variables with respect to
the response variable. The objective is to discover hidden relationships
between the independent and response variables and use those findings in
the model building process.

``` r
# First, we will explore the relationship between the target and numerical variables.
p1 <- ggplot(day) +geom_point(aes(x = temp, y = cnt), colour = "violet") + labs(title = "Normalized Temperature vs Total Rental Bikes")
p2 <- ggplot(day) +geom_point(aes(x = atemp, y = cnt), colour = "#FF99CC") +labs(title = "Normalized Feeling Temperature vs Total Rental Bikes")
p3 <- ggplot(day) +geom_point(aes(x = hum, y = cnt), colour = "pink") + labs(title = "Normalized Humidity vs Total rental Bikes")
p4 <- ggplot(day) +geom_point(aes(x = windspeed, y = cnt), colour = "#FF66CC") +labs(title= "Normalized Windspeed vs Total rental Bikes")
gridExtra::grid.arrange(p1, p2, p3, p4, nrow = 2)
```

![](Monday_files/figure-gfm/bi_var_num-1.png)<!-- -->

Observations:  
\* There appears to be a positive linear relationship between `cnt` ,
`temp`, and `atemp`.

  - There is also a weak relationship between `cnt`, `hum`, and
    `windspeed`.

<!-- end list -->

``` r
# Now we'll visualize the relationship between the target and categorical variables.
# Instead of using a boxplot, I will use a violin plot which is the blend of both a boxplot and density plot
g1 <- ggplot(day) + geom_col(aes(x = yr, y = cnt, fill = season))+theme_bw()
g2 <- ggplot(day) + geom_violin(aes(x = yr, y = cnt))+theme_bw()
g3 <- ggplot(day) + geom_col(aes(x = mnth, y = cnt, fill = season))+theme_bw() 
g4 <- ggplot(day) + geom_col(aes(x = holiday, y = cnt, fill = season)) + theme_bw() 
g6 <- ggplot(day) + geom_col(aes(x = workingday, y = cnt, fill = season))
g7 <- ggplot(day) + geom_col(aes(x = weathersit, y = cnt, fill = season))
gridExtra::grid.arrange(g1, g2, g3, nrow = 2)
```

![](Monday_files/figure-gfm/bivar_cat-1.png)<!-- -->

``` r
gridExtra::grid.arrange(g4, g6, g7, nrow = 2)
```

![](Monday_files/figure-gfm/bivar_cat-2.png)<!-- --> Observations:  
\* The total bike rental count is higher in 2012 than 2011.

  - During workingday, the bike rental counts quite the highest compared
    to during no working day for different seasons.

  - During clear,partly cloudy weather, the bike rental count is highest
    and the second highest is during mist cloudy weather and followed by
    third highest during light snow and light rain weather.

  - The highest bike rental count was during the summer and lowest in
    the winter.

## Correlation Matrix

Correlation matrix helps us to understand the linear relationship
between variables.

``` r
day_c <- day[ , c(10:14)]
round(cor(day_c), 2)
```

    ##            temp atemp   hum windspeed   cnt
    ## temp       1.00  1.00  0.19     -0.03  0.65
    ## atemp      1.00  1.00  0.22     -0.06  0.67
    ## hum        0.19  0.22  1.00     -0.42  0.00
    ## windspeed -0.03 -0.06 -0.42      1.00 -0.17
    ## cnt        0.65  0.67  0.00     -0.17  1.00

From the above matrix, we can see that `temp` and `atemp` are highly
correlated. So we only need to include one of these variables in the
model to prevent multicollinearity. We will also transform the humidity
and windspeed variable.

``` r
day <- mutate(day, log_hum = log(day$hum+1))
day <- mutate(day, log_ws = log(day$windspeed + 1))

#Remove irrelevant variables
day <- select(day, -weekday,-holiday,-workingday,-dteday,-temp, -instant)
```

## Model Building

First we split the data into train and test sets.

``` r
set.seed(23)
dayIndex<- createDataPartition(day$cnt, p = 0.7, list=FALSE)
dayTrain <- day[dayIndex, ]
dayTest <- day[-dayIndex, ]

# Build a tree-based model using loocv;
fitTree <- train(cnt~ ., data = dayTrain, method = "rpart", 
              preProcess = c("center", "scale"), 
              trControl = trainControl(method = "loocv", number = 10), tuneGrid = NULL)
```

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in nominalTrainWorkflow(x = x, y =
    ## y, wts = weights, info = trainInfo, : There
    ## were missing values in resampled performance
    ## measures.

``` r
# Display information from the tree fit
fitTree$results
```

    ##           cp     RMSE Rsquared      MAE
    ## 1 0.08349215 1028.982      NaN 1028.982
    ## 2 0.14263106 1398.644      NaN 1398.644
    ## 3 0.39843550 1873.531      NaN 1873.531
    ##      RMSESD RsquaredSD     MAESD
    ## 1 1027.3639         NA 1027.3639
    ## 2  919.1445         NA  919.1445
    ## 3  985.4040         NA  985.4040

``` r
# Build a boosted tree model using cv
fitBoost <- train(cnt~., data = dayTrain, method = "gbm", 
              preProcess = c("center", "scale"), 
              trControl = trainControl(method = "cv", number = 10), 
              tuneGrid = NULL)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2969751.8783             nan     0.1000 211751.2220
    ##      2  2774056.8289             nan     0.1000 197399.7152
    ##      3  2620806.4927             nan     0.1000 71085.0237
    ##      4  2457671.7306             nan     0.1000 132754.7676
    ##      5  2313072.9923             nan     0.1000 84602.7506
    ##      6  2178201.6536             nan     0.1000 125520.2138
    ##      7  2074009.8128             nan     0.1000 102066.5850
    ##      8  1973744.8253             nan     0.1000 75715.9708
    ##      9  1869952.3876             nan     0.1000 81197.5231
    ##     10  1814845.8699             nan     0.1000 40608.0502
    ##     20  1266659.0368             nan     0.1000 4023.5069
    ##     40   933484.7589             nan     0.1000 -14381.5879
    ##     60   783298.6360             nan     0.1000 -17488.5202
    ##     80   703922.4153             nan     0.1000 -14311.6082
    ##    100   647730.2415             nan     0.1000 -9124.5729
    ##    120   603231.6850             nan     0.1000 -9432.6922
    ##    140   540716.0902             nan     0.1000 -9422.1548
    ##    150   530469.0060             nan     0.1000 -8064.1924
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2881983.9227             nan     0.1000 178133.6200
    ##      2  2684394.4102             nan     0.1000 54304.7916
    ##      3  2461722.2929             nan     0.1000 182848.2642
    ##      4  2300536.3428             nan     0.1000 84561.8164
    ##      5  2167814.0421             nan     0.1000 104868.2714
    ##      6  2063438.3822             nan     0.1000 107369.2191
    ##      7  1887500.4609             nan     0.1000 128119.7957
    ##      8  1795385.5054             nan     0.1000 71283.9991
    ##      9  1674421.9219             nan     0.1000 103803.9331
    ##     10  1567404.9269             nan     0.1000 87546.6456
    ##     20  1146846.8885             nan     0.1000 22344.3848
    ##     40   748951.8666             nan     0.1000 10822.7475
    ##     60   645049.2622             nan     0.1000 -10983.7726
    ##     80   546173.2888             nan     0.1000 -4565.2680
    ##    100   489945.4763             nan     0.1000 -3454.3663
    ##    120   442821.7623             nan     0.1000 -2195.3479
    ##    140   388923.6432             nan     0.1000 -3691.8375
    ##    150   377684.2245             nan     0.1000 -6407.4829
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2940954.1339             nan     0.1000 230969.8711
    ##      2  2717802.5865             nan     0.1000 150578.3499
    ##      3  2536146.3490             nan     0.1000 132505.6462
    ##      4  2308192.7358             nan     0.1000 176341.5586
    ##      5  2090516.1224             nan     0.1000 164901.7373
    ##      6  1913068.1230             nan     0.1000 158038.9857
    ##      7  1816493.4927             nan     0.1000 82578.5656
    ##      8  1725539.0896             nan     0.1000 61097.3357
    ##      9  1668023.8581             nan     0.1000 28409.5980
    ##     10  1608339.2680             nan     0.1000 47362.3482
    ##     20  1061905.5422             nan     0.1000 -965.5272
    ##     40   778925.1255             nan     0.1000 -11567.2128
    ##     60   641007.1082             nan     0.1000 -16978.1709
    ##     80   576419.7399             nan     0.1000  494.3100
    ##    100   520635.5476             nan     0.1000 -4344.3766
    ##    120   469626.5430             nan     0.1000 -6167.3752
    ##    140   417193.3982             nan     0.1000 -4247.1993
    ##    150   406596.8539             nan     0.1000 -5065.2143

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in (function (x, y, offset = NULL, misc
    ## = NULL, distribution = "bernoulli", : variable
    ## 17: weathersit3 has no variation.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2823380.7790             nan     0.1000 127424.8275
    ##      2  2631240.1499             nan     0.1000 222807.9907
    ##      3  2368581.7023             nan     0.1000 199539.3927
    ##      4  2244665.8041             nan     0.1000 124756.7384
    ##      5  2071420.2630             nan     0.1000 178414.0835
    ##      6  1892755.3924             nan     0.1000 103451.7302
    ##      7  1794366.5619             nan     0.1000 92652.7004
    ##      8  1722952.1679             nan     0.1000 66042.1024
    ##      9  1601003.8605             nan     0.1000 97675.7204
    ##     10  1515821.7371             nan     0.1000 70795.4820
    ##     20  1034666.1984             nan     0.1000 -17512.9005
    ##     40   656140.1272             nan     0.1000 -2175.6513
    ##     60   483500.2053             nan     0.1000 2070.5864
    ##     80   391541.4308             nan     0.1000 -8843.5575
    ##    100   338635.4623             nan     0.1000 -5759.9232
    ##    120   311785.6943             nan     0.1000 -1254.0297
    ##    140   280881.0799             nan     0.1000 -4659.1999
    ##    150   273558.1215             nan     0.1000 -3997.3616

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in (function (x, y, offset = NULL, misc
    ## = NULL, distribution = "bernoulli", : variable
    ## 17: weathersit3 has no variation.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2797521.2307             nan     0.1000 150580.2662
    ##      2  2575918.1365             nan     0.1000 148774.6677
    ##      3  2310065.1990             nan     0.1000 219071.8477
    ##      4  2025602.4289             nan     0.1000 277173.6628
    ##      5  1900522.7344             nan     0.1000 111708.1887
    ##      6  1685293.7623             nan     0.1000 193298.1707
    ##      7  1589015.1079             nan     0.1000 65898.4095
    ##      8  1484184.7026             nan     0.1000 93071.2359
    ##      9  1404852.6518             nan     0.1000 45253.9310
    ##     10  1278511.2475             nan     0.1000 107570.4285
    ##     20   761633.9588             nan     0.1000 5024.0083
    ##     40   462909.1787             nan     0.1000 -9322.9586
    ##     60   340059.5807             nan     0.1000 -624.6036
    ##     80   272055.1761             nan     0.1000 2438.1567
    ##    100   246999.1823             nan     0.1000 -4054.4390
    ##    120   211161.4864             nan     0.1000 -2191.5244
    ##    140   190672.9392             nan     0.1000 -6800.7757
    ##    150   183823.8607             nan     0.1000 -3305.6228

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in (function (x, y, offset = NULL, misc
    ## = NULL, distribution = "bernoulli", : variable
    ## 17: weathersit3 has no variation.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2684117.5523             nan     0.1000 387423.0286
    ##      2  2487393.9252             nan     0.1000 209189.6875
    ##      3  2201073.8846             nan     0.1000 277519.6520
    ##      4  1974110.6317             nan     0.1000 220520.7842
    ##      5  1813376.8770             nan     0.1000 168660.4849
    ##      6  1624194.0987             nan     0.1000 146072.8665
    ##      7  1507432.8969             nan     0.1000 123885.8204
    ##      8  1418130.6518             nan     0.1000 53399.3602
    ##      9  1376145.7972             nan     0.1000 45454.6573
    ##     10  1285194.4062             nan     0.1000 82753.1865
    ##     20   839704.0101             nan     0.1000 -2751.8726
    ##     40   514700.7772             nan     0.1000 1995.9014
    ##     60   362436.3044             nan     0.1000 -3339.1905
    ##     80   300178.6426             nan     0.1000  278.3439
    ##    100   238424.8363             nan     0.1000 -696.3250
    ##    120   204670.6547             nan     0.1000 -1146.2008
    ##    140   184179.3728             nan     0.1000 -2250.9686
    ##    150   176531.6151             nan     0.1000 -4432.8686
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3044487.2526             nan     0.1000 245403.9084
    ##      2  2820930.4551             nan     0.1000 204610.2845
    ##      3  2689003.6836             nan     0.1000 97831.9427
    ##      4  2533873.9728             nan     0.1000 152878.0364
    ##      5  2352623.5517             nan     0.1000 119495.5300
    ##      6  2164469.5744             nan     0.1000 153788.5394
    ##      7  2038850.9268             nan     0.1000 61744.2555
    ##      8  1911042.9395             nan     0.1000 100107.7956
    ##      9  1814387.6597             nan     0.1000 87814.1628
    ##     10  1720831.0353             nan     0.1000 69501.1712
    ##     20  1188423.1745             nan     0.1000 31037.6510
    ##     40   877426.8653             nan     0.1000 -4785.1349
    ##     60   728526.4561             nan     0.1000 -10393.3231
    ##     80   664740.4553             nan     0.1000 -5763.5160
    ##    100   623082.0865             nan     0.1000 -11702.1093
    ##    120   595574.9216             nan     0.1000 -3712.7891
    ##    140   564432.8344             nan     0.1000 -11347.8710
    ##    150   550312.0667             nan     0.1000 -7471.0544
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3072587.2321             nan     0.1000 240190.0676
    ##      2  2713665.0125             nan     0.1000 293504.2214
    ##      3  2430976.4389             nan     0.1000 253540.8683
    ##      4  2267932.3021             nan     0.1000 51415.8862
    ##      5  2121342.4925             nan     0.1000 139007.7483
    ##      6  2037486.5200             nan     0.1000 73090.4395
    ##      7  1915015.4503             nan     0.1000 118858.6290
    ##      8  1737759.5588             nan     0.1000 118471.7843
    ##      9  1642557.5115             nan     0.1000 50696.2364
    ##     10  1559355.7478             nan     0.1000 51324.8888
    ##     20   989432.5158             nan     0.1000 17845.7223
    ##     40   689799.1603             nan     0.1000 -7753.5948
    ##     60   583594.6063             nan     0.1000 -15760.0472
    ##     80   508868.4461             nan     0.1000 -7146.0583
    ##    100   460817.3351             nan     0.1000 -7795.4891
    ##    120   419916.1947             nan     0.1000  245.0999
    ##    140   394291.3249             nan     0.1000  930.6658
    ##    150   392966.4722             nan     0.1000 -4560.7875
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3026088.1595             nan     0.1000 158882.9194
    ##      2  2815875.4983             nan     0.1000 179693.4666
    ##      3  2647681.2187             nan     0.1000 197436.0755
    ##      4  2518468.0787             nan     0.1000 138582.7463
    ##      5  2243123.2762             nan     0.1000 211304.6587
    ##      6  2024275.9339             nan     0.1000 193175.4387
    ##      7  1913316.4611             nan     0.1000 90171.4308
    ##      8  1733990.2364             nan     0.1000 137293.1811
    ##      9  1648606.7743             nan     0.1000 52416.5754
    ##     10  1575695.8483             nan     0.1000 56526.7388
    ##     20  1107086.8150             nan     0.1000 11846.1337
    ##     40   712832.3159             nan     0.1000 -7898.5227
    ##     60   609364.8770             nan     0.1000 -5203.7286
    ##     80   536840.2914             nan     0.1000 -6149.2518
    ##    100   485671.4836             nan     0.1000 -12787.9558
    ##    120   445974.9531             nan     0.1000 -4937.4240
    ##    140   399412.0612             nan     0.1000 -2370.3201
    ##    150   388463.5526             nan     0.1000 -6898.3020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3029863.2910             nan     0.1000 188693.2424
    ##      2  2753972.1657             nan     0.1000 173050.5786
    ##      3  2581771.1810             nan     0.1000 114578.7972
    ##      4  2473627.1803             nan     0.1000 96844.4142
    ##      5  2288955.3629             nan     0.1000 195329.0489
    ##      6  2160204.5479             nan     0.1000 104545.0029
    ##      7  2038882.3909             nan     0.1000 117174.6671
    ##      8  1941585.3620             nan     0.1000 72009.9543
    ##      9  1843936.2344             nan     0.1000 19435.8978
    ##     10  1782413.4238             nan     0.1000 38796.8495
    ##     20  1244319.5253             nan     0.1000 -22325.1344
    ##     40   871303.7205             nan     0.1000 -7579.7183
    ##     60   724530.6989             nan     0.1000 -4314.6467
    ##     80   660366.2173             nan     0.1000 -6190.5908
    ##    100   619974.5547             nan     0.1000 -3364.6013
    ##    120   606364.6762             nan     0.1000 -19184.0604
    ##    140   571603.8142             nan     0.1000 -6869.8364
    ##    150   561450.3532             nan     0.1000 -3476.6407
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2854304.6120             nan     0.1000 341910.9405
    ##      2  2531236.0966             nan     0.1000 287976.8627
    ##      3  2266059.8663             nan     0.1000 209699.8470
    ##      4  2073764.0412             nan     0.1000 121819.1768
    ##      5  1851516.7048             nan     0.1000 129044.8813
    ##      6  1768687.7740             nan     0.1000 53144.6462
    ##      7  1703180.2033             nan     0.1000 79528.3224
    ##      8  1590238.5720             nan     0.1000 42637.9696
    ##      9  1483138.2354             nan     0.1000 89875.8468
    ##     10  1434582.3786             nan     0.1000 8740.3771
    ##     20   979367.2324             nan     0.1000 11798.3761
    ##     40   699019.1446             nan     0.1000 -14360.9147
    ##     60   588995.1862             nan     0.1000 -14530.6462
    ##     80   533908.9520             nan     0.1000 -5086.6823
    ##    100   460839.7652             nan     0.1000 -9783.8065
    ##    120   429384.5344             nan     0.1000 -2368.6951
    ##    140   368632.0411             nan     0.1000 -4792.3268
    ##    150   354760.6160             nan     0.1000 -14838.7918
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3120428.8857             nan     0.1000 94123.4095
    ##      2  2810200.2322             nan     0.1000 313907.4996
    ##      3  2524061.1585             nan     0.1000 274040.8882
    ##      4  2417916.8690             nan     0.1000 83057.9772
    ##      5  2208314.9213             nan     0.1000 197124.5991
    ##      6  2072070.9296             nan     0.1000 123630.8564
    ##      7  1889763.5986             nan     0.1000 97944.6883
    ##      8  1779730.3614             nan     0.1000 70698.8413
    ##      9  1645208.3425             nan     0.1000 115968.2389
    ##     10  1567461.9965             nan     0.1000 32549.1843
    ##     20  1041544.4397             nan     0.1000 8609.2426
    ##     40   691148.7769             nan     0.1000 -6006.5373
    ##     60   580572.2575             nan     0.1000 -6677.5080
    ##     80   513466.8397             nan     0.1000 -17605.1978
    ##    100   465913.6497             nan     0.1000 -4357.7493
    ##    120   420688.8581             nan     0.1000 -684.0241
    ##    140   377533.4906             nan     0.1000 -4128.4999
    ##    150   350568.0834             nan     0.1000 -3349.9129
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3226719.8090             nan     0.1000 123002.7173
    ##      2  3097130.7000             nan     0.1000 136399.2355
    ##      3  2939958.7464             nan     0.1000 217039.2624
    ##      4  2687844.1595             nan     0.1000 154020.1856
    ##      5  2534726.4981             nan     0.1000 123505.1302
    ##      6  2357633.1356             nan     0.1000 147520.0327
    ##      7  2243278.9111             nan     0.1000 69501.3470
    ##      8  2128780.1208             nan     0.1000 133787.6069
    ##      9  2085626.4635             nan     0.1000 -3830.0279
    ##     10  1990901.7180             nan     0.1000 93291.6473
    ##     20  1302191.5221             nan     0.1000 1846.1558
    ##     40   923027.9837             nan     0.1000 -7832.0643
    ##     60   784517.3242             nan     0.1000 -9285.3259
    ##     80   688334.5761             nan     0.1000 -127.6280
    ##    100   642284.0833             nan     0.1000 -3548.7139
    ##    120   598801.1370             nan     0.1000 -18845.4741
    ##    140   559185.8810             nan     0.1000 -10561.6552
    ##    150   542460.8157             nan     0.1000  159.7465
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2947789.3412             nan     0.1000 312214.3555
    ##      2  2702311.4525             nan     0.1000 259214.1233
    ##      3  2499796.2586             nan     0.1000 200834.5733
    ##      4  2259067.5280             nan     0.1000 144828.8510
    ##      5  2095264.9173             nan     0.1000 90941.0335
    ##      6  1980580.0403             nan     0.1000 67291.5834
    ##      7  1914188.8819             nan     0.1000 47029.0650
    ##      8  1819890.6594             nan     0.1000 72560.9745
    ##      9  1705516.4489             nan     0.1000 102190.4278
    ##     10  1630947.3911             nan     0.1000 36507.5300
    ##     20  1137255.7469             nan     0.1000 45445.3179
    ##     40   807647.9474             nan     0.1000 -5466.6903
    ##     60   663078.9506             nan     0.1000 -10827.5954
    ##     80   570044.9879             nan     0.1000 -10943.2091
    ##    100   506336.8680             nan     0.1000 -1600.5921
    ##    120   457769.2083             nan     0.1000 1267.5188
    ##    140   408856.5007             nan     0.1000 -6668.6914
    ##    150   380750.0843             nan     0.1000 -7845.8072
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3077511.3387             nan     0.1000 174315.0464
    ##      2  2757893.0872             nan     0.1000 295123.3821
    ##      3  2501743.8996             nan     0.1000 245783.1452
    ##      4  2378576.2796             nan     0.1000 134315.2979
    ##      5  2257273.0139             nan     0.1000 95174.9689
    ##      6  2050605.0128             nan     0.1000 137408.8005
    ##      7  1961800.7858             nan     0.1000 46983.7271
    ##      8  1835907.0245             nan     0.1000 122150.1225
    ##      9  1715054.7559             nan     0.1000 80300.0473
    ##     10  1621694.5816             nan     0.1000 10588.0762
    ##     20  1040908.0933             nan     0.1000 16577.2147
    ##     40   743261.2971             nan     0.1000 -19000.5198
    ##     60   601316.9550             nan     0.1000 -7485.6321
    ##     80   523034.7836             nan     0.1000 -7369.7554
    ##    100   449500.5126             nan     0.1000 -8945.2277
    ##    120   405228.1107             nan     0.1000 -3271.1123
    ##    140   373172.5619             nan     0.1000 -4637.1466
    ##    150   353420.2033             nan     0.1000 -4911.1372
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3204475.9075             nan     0.1000 160794.5581
    ##      2  2997672.3216             nan     0.1000 163128.3998
    ##      3  2766588.0876             nan     0.1000 206828.7110
    ##      4  2569300.6751             nan     0.1000 135289.8285
    ##      5  2426377.0390             nan     0.1000 141452.6758
    ##      6  2334871.2164             nan     0.1000 72848.4475
    ##      7  2212094.0470             nan     0.1000 74363.6542
    ##      8  2128111.4138             nan     0.1000 64149.0723
    ##      9  2002020.0179             nan     0.1000 117192.9538
    ##     10  1974790.2113             nan     0.1000 -22736.6142
    ##     20  1429416.4692             nan     0.1000 7414.7885
    ##     40   937687.7892             nan     0.1000 -13653.7566
    ##     60   754577.8763             nan     0.1000 -5087.4915
    ##     80   682444.2704             nan     0.1000 -19611.5289
    ##    100   641687.0548             nan     0.1000 -7145.1604
    ##    120   606539.2107             nan     0.1000 -5439.8711
    ##    140   583098.1262             nan     0.1000 -5520.3983
    ##    150   571105.9115             nan     0.1000 -7421.9660
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2975981.8403             nan     0.1000 333525.2859
    ##      2  2646758.1076             nan     0.1000 238080.3430
    ##      3  2420824.2161             nan     0.1000 228053.2289
    ##      4  2257402.3648             nan     0.1000 156595.6379
    ##      5  2165043.8615             nan     0.1000 94116.9148
    ##      6  2065663.0232             nan     0.1000 69957.8729
    ##      7  1894271.1224             nan     0.1000 87370.0307
    ##      8  1775258.6832             nan     0.1000 113160.9159
    ##      9  1656948.3565             nan     0.1000 53232.4619
    ##     10  1613537.7412             nan     0.1000 24035.7912
    ##     20  1117709.8420             nan     0.1000 27801.5223
    ##     40   730780.5979             nan     0.1000 -8562.6431
    ##     60   619466.3008             nan     0.1000  164.8353
    ##     80   570820.7024             nan     0.1000 -10311.2469
    ##    100   528053.2521             nan     0.1000 -21254.7299
    ##    120   481733.5028             nan     0.1000 -16152.8310
    ##    140   448162.6169             nan     0.1000 -4142.8786
    ##    150   420550.6042             nan     0.1000 -3707.1004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3082459.7363             nan     0.1000 271593.3517
    ##      2  2837094.1783             nan     0.1000 176486.6535
    ##      3  2697862.1106             nan     0.1000 151994.1903
    ##      4  2494861.9796             nan     0.1000 95364.1853
    ##      5  2321342.9990             nan     0.1000 122850.9193
    ##      6  2146620.6010             nan     0.1000 176858.8204
    ##      7  2047398.2594             nan     0.1000 88824.1705
    ##      8  1932361.2671             nan     0.1000 95069.3248
    ##      9  1788437.4669             nan     0.1000 98652.6910
    ##     10  1730197.2512             nan     0.1000 62554.8160
    ##     20  1095221.8010             nan     0.1000 21051.1347
    ##     40   749728.4407             nan     0.1000 11477.3491
    ##     60   659675.8938             nan     0.1000 -15186.2816
    ##     80   571454.0776             nan     0.1000 -624.7818
    ##    100   504714.6819             nan     0.1000 -17349.1089
    ##    120   460574.4445             nan     0.1000 -1011.5716
    ##    140   406531.1770             nan     0.1000 -7546.7694
    ##    150   382635.3450             nan     0.1000 -11767.0635
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3063226.7454             nan     0.1000 223699.8518
    ##      2  2890328.9215             nan     0.1000 138950.2850
    ##      3  2695233.4533             nan     0.1000 165142.8142
    ##      4  2518030.3584             nan     0.1000 145444.2233
    ##      5  2344039.4016             nan     0.1000 62057.4855
    ##      6  2257253.6798             nan     0.1000 79548.1109
    ##      7  2123429.7552             nan     0.1000 116487.5091
    ##      8  2033790.9363             nan     0.1000 92681.6092
    ##      9  1945604.4645             nan     0.1000 47944.4057
    ##     10  1883591.1236             nan     0.1000 50212.9001
    ##     20  1357094.5749             nan     0.1000 22296.2748
    ##     40   931015.7799             nan     0.1000 -3794.9262
    ##     60   753696.4636             nan     0.1000 5611.5360
    ##     80   668653.2226             nan     0.1000 -11907.5618
    ##    100   623158.0125             nan     0.1000 -3403.7201
    ##    120   591611.5300             nan     0.1000 -5442.8782
    ##    140   552872.9692             nan     0.1000 -9432.6267
    ##    150   544997.3419             nan     0.1000 -5259.5710
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3058997.5609             nan     0.1000 157689.1628
    ##      2  2904493.2554             nan     0.1000 128627.1769
    ##      3  2667856.6566             nan     0.1000 183543.3241
    ##      4  2534300.6475             nan     0.1000 104153.6507
    ##      5  2350662.4527             nan     0.1000 98805.0805
    ##      6  2163836.6458             nan     0.1000 114695.3007
    ##      7  1951617.4935             nan     0.1000 124416.0886
    ##      8  1884203.9411             nan     0.1000 80845.5374
    ##      9  1810094.8041             nan     0.1000 75148.0606
    ##     10  1737043.8353             nan     0.1000 60107.3098
    ##     20  1169918.2167             nan     0.1000 10653.8705
    ##     40   822006.9227             nan     0.1000 -9485.6268
    ##     60   700725.9761             nan     0.1000 -18434.9295
    ##     80   577830.8396             nan     0.1000 -12127.6907
    ##    100   523033.9173             nan     0.1000 -7096.8761
    ##    120   438539.5944             nan     0.1000 -4508.0042
    ##    140   399357.7245             nan     0.1000 -6594.7906
    ##    150   386516.0074             nan     0.1000 -8541.4528
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2933736.5001             nan     0.1000 306365.5374
    ##      2  2692610.0955             nan     0.1000 277039.7455
    ##      3  2428341.5696             nan     0.1000 243087.7529
    ##      4  2246739.4257             nan     0.1000 174217.3316
    ##      5  2145636.0332             nan     0.1000 101049.4464
    ##      6  2051266.0855             nan     0.1000 53373.1148
    ##      7  1884885.9939             nan     0.1000 147441.7801
    ##      8  1786198.0212             nan     0.1000 68556.4121
    ##      9  1728680.0431             nan     0.1000 28640.6867
    ##     10  1584789.6064             nan     0.1000 105911.8969
    ##     20  1138279.4130             nan     0.1000  485.0375
    ##     40   730531.6957             nan     0.1000 -5550.9683
    ##     60   604268.0759             nan     0.1000 -17958.5781
    ##     80   544237.3715             nan     0.1000 -13200.1416
    ##    100   475610.2127             nan     0.1000 -6690.9720
    ##    120   431971.1603             nan     0.1000 -8321.7104
    ##    140   399627.9532             nan     0.1000 -5210.8757
    ##    150   384984.0053             nan     0.1000 -4098.7365
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3080133.8101             nan     0.1000 227924.2243
    ##      2  2857178.3020             nan     0.1000 176012.1106
    ##      3  2679341.0540             nan     0.1000 161234.5079
    ##      4  2498308.7948             nan     0.1000 168278.6477
    ##      5  2351347.0889             nan     0.1000 72183.0080
    ##      6  2222154.3350             nan     0.1000 133036.3279
    ##      7  2117291.3553             nan     0.1000 77791.7385
    ##      8  2032383.1628             nan     0.1000 108566.9693
    ##      9  1943841.3709             nan     0.1000 83089.7585
    ##     10  1857225.7424             nan     0.1000 79150.7947
    ##     20  1262205.2573             nan     0.1000 11922.5734
    ##     40   889927.2696             nan     0.1000 -6732.8266
    ##     60   771665.0847             nan     0.1000  658.4938
    ##     80   684457.2251             nan     0.1000 -12764.8271
    ##    100   636133.9357             nan     0.1000 -3662.6844
    ##    120   602552.3334             nan     0.1000 -4061.8387
    ##    140   576945.5772             nan     0.1000 -6750.0058
    ##    150   560127.0525             nan     0.1000 -7936.7996
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2937638.6094             nan     0.1000 294800.2277
    ##      2  2693258.6991             nan     0.1000 191615.2510
    ##      3  2500374.1260             nan     0.1000 136634.0260
    ##      4  2330014.3102             nan     0.1000 130383.6644
    ##      5  2209907.6018             nan     0.1000 95155.0210
    ##      6  2086590.4770             nan     0.1000 120810.0803
    ##      7  1930365.7448             nan     0.1000 98086.3960
    ##      8  1812348.4988             nan     0.1000 86457.7703
    ##      9  1739518.9602             nan     0.1000 60159.7580
    ##     10  1668418.3557             nan     0.1000 46322.6421
    ##     20  1263081.9352             nan     0.1000 -23809.4195
    ##     40   765508.4987             nan     0.1000 -9602.6963
    ##     60   641442.1661             nan     0.1000 -9985.6482
    ##     80   560178.0672             nan     0.1000 -5906.4425
    ##    100   498090.2418             nan     0.1000 -2248.4150
    ##    120   442005.0414             nan     0.1000 -6807.5838
    ##    140   394107.0544             nan     0.1000 -7002.9095
    ##    150   376501.5680             nan     0.1000 -16176.2855
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3107647.1711             nan     0.1000 120871.6342
    ##      2  2747314.5133             nan     0.1000 342152.3751
    ##      3  2501642.7623             nan     0.1000 157814.6040
    ##      4  2293854.9399             nan     0.1000 219399.6204
    ##      5  2075977.1558             nan     0.1000 162283.5237
    ##      6  1947613.6376             nan     0.1000 129450.1201
    ##      7  1901424.5069             nan     0.1000 3002.6688
    ##      8  1766188.6469             nan     0.1000 83574.3204
    ##      9  1656431.3684             nan     0.1000 66269.9983
    ##     10  1589646.7042             nan     0.1000 44169.9310
    ##     20  1038308.6013             nan     0.1000 16782.2716
    ##     40   719421.8025             nan     0.1000 -38289.7498
    ##     60   583752.6050             nan     0.1000 -8200.3160
    ##     80   498253.0810             nan     0.1000 -9713.6020
    ##    100   439743.2421             nan     0.1000 -1599.0874
    ##    120   385328.7897             nan     0.1000 -2497.7357
    ##    140   346408.3553             nan     0.1000 -2100.9770
    ##    150   329650.5531             nan     0.1000 -5850.7673
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3163482.2933             nan     0.1000 160612.3045
    ##      2  2959080.0872             nan     0.1000 196475.9110
    ##      3  2782511.4473             nan     0.1000 178989.4108
    ##      4  2644246.2311             nan     0.1000 151662.6960
    ##      5  2496653.2570             nan     0.1000 114186.9936
    ##      6  2321882.7006             nan     0.1000 147378.0854
    ##      7  2147676.5094             nan     0.1000 108792.1992
    ##      8  2034696.8015             nan     0.1000 84129.6015
    ##      9  1956160.5106             nan     0.1000 92757.5366
    ##     10  1863875.8577             nan     0.1000 72965.5164
    ##     20  1331873.7098             nan     0.1000 36563.2852
    ##     40   864944.6018             nan     0.1000 -5961.7951
    ##     60   684411.3001             nan     0.1000 -8543.6192
    ##     80   615584.4601             nan     0.1000 -11630.1061
    ##    100   580667.0118             nan     0.1000 -2946.7366
    ##    120   545021.7786             nan     0.1000 -4585.6286
    ##    140   521255.2865             nan     0.1000 -16901.9991
    ##    150   507447.4094             nan     0.1000 -8243.6333
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3065568.0362             nan     0.1000 324441.8282
    ##      2  2777759.4221             nan     0.1000 295101.0316
    ##      3  2587554.8759             nan     0.1000 140830.6580
    ##      4  2445020.0574             nan     0.1000 141885.5378
    ##      5  2221854.1488             nan     0.1000 183862.8135
    ##      6  2075266.9799             nan     0.1000 109319.7490
    ##      7  1928872.9360             nan     0.1000 87745.8573
    ##      8  1761965.2887             nan     0.1000 88442.3819
    ##      9  1651209.5927             nan     0.1000 97584.6049
    ##     10  1594713.6356             nan     0.1000 42307.2517
    ##     20  1070566.8391             nan     0.1000 -681.0539
    ##     40   733891.2896             nan     0.1000 16479.5106
    ##     60   619923.5025             nan     0.1000 -7987.9995
    ##     80   552830.9755             nan     0.1000 -1034.9913
    ##    100   500011.4217             nan     0.1000 -5069.8068
    ##    120   455463.8712             nan     0.1000 -6627.2711
    ##    140   400833.1854             nan     0.1000 -6235.2947
    ##    150   377857.8556             nan     0.1000 -9195.8899
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3076079.8661             nan     0.1000 382339.0430
    ##      2  2741154.8911             nan     0.1000 317124.8501
    ##      3  2565468.4421             nan     0.1000 121589.5630
    ##      4  2312825.6222             nan     0.1000 194673.1139
    ##      5  2125769.3385             nan     0.1000 125550.9860
    ##      6  1966832.7399             nan     0.1000 95847.9427
    ##      7  1852332.9124             nan     0.1000 60015.9985
    ##      8  1787709.1082             nan     0.1000 25441.5296
    ##      9  1680899.0979             nan     0.1000 90794.9663
    ##     10  1598514.6247             nan     0.1000 67464.7815
    ##     20  1046797.2853             nan     0.1000 20875.5591
    ##     40   691792.9763             nan     0.1000 -9753.2348
    ##     60   583972.9511             nan     0.1000 -2931.6593
    ##     80   503667.6994             nan     0.1000 -9362.6214
    ##    100   454378.5945             nan     0.1000 -9628.5295
    ##    120   428660.1394             nan     0.1000 -15247.8942
    ##    140   379342.7327             nan     0.1000 -4138.9338
    ##    150   370711.6044             nan     0.1000 -6314.8308
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3096325.2215             nan     0.1000 222957.1158
    ##      2  2853715.7804             nan     0.1000 218374.4921
    ##      3  2670961.0059             nan     0.1000 109798.0981
    ##      4  2426526.1304             nan     0.1000 101508.0469
    ##      5  2281568.9667             nan     0.1000 115778.1889
    ##      6  2154368.0411             nan     0.1000 85238.3967
    ##      7  2061501.1084             nan     0.1000 50894.2224
    ##      8  1958812.9104             nan     0.1000 95467.5855
    ##      9  1861975.6373             nan     0.1000 21606.6988
    ##     10  1756668.6639             nan     0.1000 108799.9831
    ##     20  1245047.0994             nan     0.1000 24147.1639
    ##     40   832998.3367             nan     0.1000 -3513.5712
    ##     60   693322.2917             nan     0.1000  903.1751
    ##     80   637110.5242             nan     0.1000 -6032.4739
    ##    100   581729.2428             nan     0.1000 1104.5975
    ##    120   540193.9765             nan     0.1000 -7178.5348
    ##    140   504084.7665             nan     0.1000 -8488.0981
    ##    150   491713.7661             nan     0.1000 -9127.5991
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3066338.8371             nan     0.1000 354223.3810
    ##      2  2828555.1953             nan     0.1000 192789.1468
    ##      3  2564858.9294             nan     0.1000 184333.0483
    ##      4  2309798.4212             nan     0.1000 164384.8634
    ##      5  2190369.9337             nan     0.1000 120283.9717
    ##      6  2063907.0990             nan     0.1000 74299.5785
    ##      7  1969353.9829             nan     0.1000 45137.3814
    ##      8  1854431.7782             nan     0.1000 59498.9356
    ##      9  1775813.9480             nan     0.1000 91117.1853
    ##     10  1643555.8879             nan     0.1000 84244.4810
    ##     20  1047196.8991             nan     0.1000 11225.3666
    ##     40   687475.5266             nan     0.1000 4767.1603
    ##     60   585182.6988             nan     0.1000 -9218.3220
    ##     80   531777.5266             nan     0.1000 -3299.9764
    ##    100   469718.8165             nan     0.1000 -11045.8334
    ##    120   428574.7955             nan     0.1000 -12928.3291
    ##    140   393758.6780             nan     0.1000 -7937.4709
    ##    150   382453.3676             nan     0.1000 -4867.8783
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3128978.2768             nan     0.1000 262664.6627
    ##      2  2916745.5839             nan     0.1000 204376.2872
    ##      3  2646535.7270             nan     0.1000 271660.5028
    ##      4  2456719.7661             nan     0.1000 136039.4151
    ##      5  2288918.2855             nan     0.1000 84541.3724
    ##      6  2078426.2691             nan     0.1000 164896.1658
    ##      7  1961491.4166             nan     0.1000 107634.7651
    ##      8  1813276.1774             nan     0.1000 137971.6615
    ##      9  1689289.9962             nan     0.1000 98749.0826
    ##     10  1614398.9453             nan     0.1000 54179.9582
    ##     20  1057827.0938             nan     0.1000 14507.3755
    ##     40   720536.7975             nan     0.1000 -2355.8390
    ##     60   604273.4716             nan     0.1000  513.8613
    ##     80   530718.1519             nan     0.1000 -5419.5320
    ##    100   465582.6151             nan     0.1000 -2942.8279
    ##    120   434493.5678             nan     0.1000 -5520.1033
    ##    140   401003.4794             nan     0.1000 -4308.6167
    ##    150   383422.8134             nan     0.1000 -9697.9699
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3102485.6838             nan     0.1000 131543.6346
    ##      2  2875753.4965             nan     0.1000 255117.2032
    ##      3  2533617.5621             nan     0.1000 244136.8197
    ##      4  2339144.3270             nan     0.1000 148520.8580
    ##      5  2156252.4819             nan     0.1000 113740.9747
    ##      6  1972327.7907             nan     0.1000 181004.8202
    ##      7  1799862.9090             nan     0.1000 149720.1109
    ##      8  1677962.8864             nan     0.1000 109025.6139
    ##      9  1596586.7347             nan     0.1000 75060.4938
    ##     10  1525016.9268             nan     0.1000 74039.6850
    ##     20  1000504.0311             nan     0.1000 41276.6716
    ##     40   652215.1729             nan     0.1000 9627.5034
    ##     60   583964.9967             nan     0.1000 -12001.5583
    ##     80   521094.3570             nan     0.1000 -6103.2167
    ##    100   487984.8246             nan     0.1000 -11589.3069
    ##    120   440213.3267             nan     0.1000 -7312.0694
    ##    140   386717.9108             nan     0.1000 -8557.0710
    ##    150   362671.1818             nan     0.1000 -1939.6783

``` r
# Display information from the boost fit
fitBoost$results
```

    ##   shrinkage interaction.depth n.minobsinnode
    ## 1       0.1                 1             10
    ## 4       0.1                 2             10
    ## 7       0.1                 3             10
    ## 2       0.1                 1             10
    ## 5       0.1                 2             10
    ## 8       0.1                 3             10
    ## 3       0.1                 1             10
    ## 6       0.1                 2             10
    ## 9       0.1                 3             10
    ##   n.trees      RMSE  Rsquared      MAE   RMSESD
    ## 1      50 1002.3177 0.7083885 768.7459 440.8178
    ## 4      50  936.1469 0.7315391 701.4452 452.8629
    ## 7      50  969.1061 0.7105741 731.9262 463.0326
    ## 2     100  948.0518 0.7284715 708.6800 500.0684
    ## 5     100  939.7427 0.7305958 685.0603 453.4424
    ## 8     100  942.3049 0.7336082 705.9563 489.7862
    ## 3     150  954.5356 0.7295462 715.5056 486.4046
    ## 6     150  926.1170 0.7454583 687.9787 445.0895
    ## 9     150  929.1026 0.7487459 696.4390 507.1643
    ##   RsquaredSD    MAESD
    ## 1  0.2357072 255.5954
    ## 4  0.2016909 224.5759
    ## 7  0.2306157 286.7485
    ## 2  0.2565736 298.0391
    ## 5  0.2114234 220.3925
    ## 8  0.2246564 271.4480
    ## 3  0.2469190 282.7059
    ## 6  0.1967603 211.7986
    ## 9  0.2282818 267.2765

Now, we make predictions on the test data sets using the best model
fits. Then we compare RMSE to determine the best model.

``` r
predTree <- predict(fitTree, newdata = select(dayTest, -cnt))
postResample(predTree, dayTest$cnt)
```

    ##         RMSE     Rsquared          MAE 
    ## 1011.0632190    0.6532397  801.8606897

``` r
boostPred <- predict(fitBoost, newdata = select(dayTest, -cnt))
postResample(boostPred, dayTest$cnt)
```

    ##        RMSE    Rsquared         MAE 
    ## 676.3616438   0.8540732 532.4379473

When we compare the two models, the boosted tree model have lower RMSE
values when applied on the test dataset.
