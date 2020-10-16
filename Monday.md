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
              trControl = trainControl(method = "loocv", number = 10), tuneGrid = data.frame(cp = 0.01:0.10))
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

    ##     cp     RMSE Rsquared      MAE   RMSESD
    ## 1 0.01 869.1544      NaN 869.1544 937.8054
    ##   RsquaredSD    MAESD
    ## 1         NA 937.8054

``` r
# Build a boosted tree model using cv
fitBoost <- train(cnt~., data = dayTrain, method = "gbm", 
              preProcess = c("center", "scale"), 
              trControl = trainControl(method = "cv", number = 10), 
              tuneGrid = expand.grid(n.trees=c(10,20),shrinkage=c(0.01,0.05),n.minobsinnode =c(3),interaction.depth=c(1,5)))
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3481869.3341             nan     0.0100 27192.9345
    ##      2  3452412.5499             nan     0.0100 30037.0243
    ##      3  3417458.6348             nan     0.0100 24336.5760
    ##      4  3383492.4472             nan     0.0100 22147.9565
    ##      5  3353030.9822             nan     0.0100 18525.2399
    ##      6  3321079.6915             nan     0.0100 20220.3608
    ##      7  3299787.6087             nan     0.0100 25222.9235
    ##      8  3279598.1247             nan     0.0100 15311.6465
    ##      9  3259022.0947             nan     0.0100 15943.9119
    ##     10  3233890.9583             nan     0.0100 21505.7565
    ##     20  3007735.2960             nan     0.0100 21251.6956
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3454998.5281             nan     0.0100 56031.3861
    ##      2  3400381.2106             nan     0.0100 54302.9141
    ##      3  3347172.0897             nan     0.0100 56152.5723
    ##      4  3292501.5755             nan     0.0100 60982.2687
    ##      5  3239786.5241             nan     0.0100 35654.7096
    ##      6  3189162.2728             nan     0.0100 48020.4682
    ##      7  3138945.8084             nan     0.0100 41259.8228
    ##      8  3093614.1283             nan     0.0100 37162.1577
    ##      9  3047000.1290             nan     0.0100 33047.2277
    ##     10  3000638.3935             nan     0.0100 40966.6324
    ##     20  2600809.9014             nan     0.0100 22565.3178
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3360965.0699             nan     0.0500 114942.5513
    ##      2  3230071.7231             nan     0.0500 124072.4764
    ##      3  3123826.0142             nan     0.0500 72973.9497
    ##      4  2991056.3940             nan     0.0500 111569.0651
    ##      5  2892353.9425             nan     0.0500 89150.5332
    ##      6  2799982.9460             nan     0.0500 99148.2023
    ##      7  2697373.5921             nan     0.0500 97144.8897
    ##      8  2674741.6728             nan     0.0500 -33449.8554
    ##      9  2570292.7743             nan     0.0500 84745.1744
    ##     10  2496845.0403             nan     0.0500 68024.6186
    ##     20  1848733.6865             nan     0.0500 36141.3321
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3219489.2203             nan     0.0500 233910.3084
    ##      2  2978803.7505             nan     0.0500 198461.2670
    ##      3  2749556.4908             nan     0.0500 195661.7058
    ##      4  2550955.2798             nan     0.0500 158967.0994
    ##      5  2389254.0993             nan     0.0500 93066.2521
    ##      6  2200671.3862             nan     0.0500 110208.6239
    ##      7  2068740.2436             nan     0.0500 128050.4158
    ##      8  1934802.1573             nan     0.0500 115604.6503
    ##      9  1793310.4701             nan     0.0500 109279.2550
    ##     10  1678699.5002             nan     0.0500 89577.4100
    ##     20   943612.7053             nan     0.0500 21290.2848
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3218893.3156             nan     0.0100 19480.7115
    ##      2  3194175.0397             nan     0.0100 25814.5348
    ##      3  3166381.6949             nan     0.0100 21537.3223
    ##      4  3137058.4871             nan     0.0100 20926.6819
    ##      5  3117324.2046             nan     0.0100 17131.7300
    ##      6  3099330.2462             nan     0.0100 17135.9493
    ##      7  3072199.0967             nan     0.0100 14929.7318
    ##      8  3048400.0911             nan     0.0100 18353.5388
    ##      9  3027519.9901             nan     0.0100 13883.7963
    ##     10  3006249.1465             nan     0.0100 11286.3247
    ##     20  2805363.1688             nan     0.0100 13732.2527
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3198853.8253             nan     0.0100 41665.3125
    ##      2  3147586.9831             nan     0.0100 43573.4221
    ##      3  3100983.1084             nan     0.0100 39738.0043
    ##      4  3059105.5836             nan     0.0100 26681.4504
    ##      5  3018108.7307             nan     0.0100 32664.3758
    ##      6  2969289.3309             nan     0.0100 36708.4228
    ##      7  2933108.9846             nan     0.0100 32775.0194
    ##      8  2894626.0118             nan     0.0100 22724.2086
    ##      9  2852529.3030             nan     0.0100 31409.3512
    ##     10  2812457.1519             nan     0.0100 37660.1392
    ##     20  2459087.7048             nan     0.0100 24829.8197
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3109763.5651             nan     0.0500 93197.0557
    ##      2  2978208.9254             nan     0.0500 97086.4215
    ##      3  2879464.3630             nan     0.0500 97409.0106
    ##      4  2806579.8449             nan     0.0500 43074.4639
    ##      5  2716246.8755             nan     0.0500 77745.0320
    ##      6  2634665.6119             nan     0.0500 83109.0462
    ##      7  2546218.5662             nan     0.0500 56895.4980
    ##      8  2463827.4084             nan     0.0500 65828.6283
    ##      9  2386688.7672             nan     0.0500 53673.5963
    ##     10  2311657.5418             nan     0.0500 51247.2011
    ##     20  1836420.9044             nan     0.0500 -9475.5432
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3032505.4352             nan     0.0500 137891.5273
    ##      2  2839698.7628             nan     0.0500 133287.4536
    ##      3  2636850.5901             nan     0.0500 166143.1316
    ##      4  2460715.6329             nan     0.0500 161555.3138
    ##      5  2258185.4189             nan     0.0500 110240.3114
    ##      6  2127614.4227             nan     0.0500 84192.6661
    ##      7  2019519.0277             nan     0.0500 72188.0297
    ##      8  1877896.1847             nan     0.0500 99647.1985
    ##      9  1769110.3408             nan     0.0500 66032.1066
    ##     10  1692768.6963             nan     0.0500 61955.3275
    ##     20  1015054.9802             nan     0.0500 27416.1199

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in (function (x, y, offset = NULL, misc
    ## = NULL, distribution = "bernoulli", : variable
    ## 17: weathersit3 has no variation.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2913039.0911             nan     0.0100 26590.6845
    ##      2  2888813.9720             nan     0.0100 19962.9483
    ##      3  2868362.3391             nan     0.0100 13926.6482
    ##      4  2847477.7191             nan     0.0100 21192.1090
    ##      5  2816413.8822             nan     0.0100 30542.6189
    ##      6  2795951.9785             nan     0.0100 11519.2362
    ##      7  2774191.5025             nan     0.0100 22051.1552
    ##      8  2749934.3331             nan     0.0100 23340.0555
    ##      9  2731486.5298             nan     0.0100 10622.7100
    ##     10  2706928.0379             nan     0.0100 17592.7164
    ##     20  2479826.3658             nan     0.0100 17265.1241

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in (function (x, y, offset = NULL, misc
    ## = NULL, distribution = "bernoulli", : variable
    ## 17: weathersit3 has no variation.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2889197.1362             nan     0.0100 44518.7243
    ##      2  2845001.9510             nan     0.0100 36232.4956
    ##      3  2802858.5880             nan     0.0100 38938.0740
    ##      4  2755836.8190             nan     0.0100 46760.6519
    ##      5  2712089.8471             nan     0.0100 37321.5660
    ##      6  2675748.0387             nan     0.0100 27160.2392
    ##      7  2628690.6243             nan     0.0100 43241.0019
    ##      8  2591187.7163             nan     0.0100 31924.0180
    ##      9  2548973.7945             nan     0.0100 39828.5164
    ##     10  2510771.7931             nan     0.0100 41407.0526
    ##     20  2159094.3181             nan     0.0100 28904.1504

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in (function (x, y, offset = NULL, misc
    ## = NULL, distribution = "bernoulli", : variable
    ## 17: weathersit3 has no variation.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2842423.9783             nan     0.0500 88107.8639
    ##      2  2728041.0224             nan     0.0500 136426.1467
    ##      3  2613415.9862             nan     0.0500 109365.2744
    ##      4  2545365.0622             nan     0.0500 53747.1041
    ##      5  2437052.0415             nan     0.0500 93920.8590
    ##      6  2344158.7246             nan     0.0500 70887.1437
    ##      7  2214873.3223             nan     0.0500 74416.4060
    ##      8  2132004.6166             nan     0.0500 64436.9737
    ##      9  2052420.4630             nan     0.0500 57204.1547
    ##     10  1976066.6244             nan     0.0500 65567.3721
    ##     20  1448147.0497             nan     0.0500 44000.8100

    ## Warning in preProcess.default(thresh = 0.95,
    ## k = 5, freqCut = 19, uniqueCut = 10, : These
    ## variables have zero variances: weathersit3

    ## Warning in (function (x, y, offset = NULL, misc
    ## = NULL, distribution = "bernoulli", : variable
    ## 17: weathersit3 has no variation.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2706366.8415             nan     0.0500 249254.0520
    ##      2  2491782.3463             nan     0.0500 172266.6050
    ##      3  2296208.6536             nan     0.0500 170744.0569
    ##      4  2108161.5078             nan     0.0500 195358.4908
    ##      5  1959824.2462             nan     0.0500 126481.1529
    ##      6  1836000.8979             nan     0.0500 96087.7453
    ##      7  1720128.4166             nan     0.0500 132974.7026
    ##      8  1598470.2156             nan     0.0500 61982.5683
    ##      9  1477135.4424             nan     0.0500 116914.5332
    ##     10  1392829.5423             nan     0.0500 57291.6326
    ##     20   780521.7339             nan     0.0500 23122.3886
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3380066.3439             nan     0.0100 25660.5632
    ##      2  3350830.8231             nan     0.0100 25333.1850
    ##      3  3327753.5001             nan     0.0100 10190.7550
    ##      4  3303230.6064             nan     0.0100 24724.8167
    ##      5  3272432.7955             nan     0.0100 21515.1431
    ##      6  3256867.3992             nan     0.0100 9823.9678
    ##      7  3229623.9544             nan     0.0100 22367.7927
    ##      8  3201024.6374             nan     0.0100 20554.1336
    ##      9  3175621.7754             nan     0.0100 26472.6939
    ##     10  3156193.0010             nan     0.0100 22177.8464
    ##     20  2959224.1157             nan     0.0100 5168.1359
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3358227.3630             nan     0.0100 50726.7760
    ##      2  3310030.3575             nan     0.0100 38608.8308
    ##      3  3265389.8871             nan     0.0100 26975.3297
    ##      4  3220404.1164             nan     0.0100 32091.0238
    ##      5  3172870.4451             nan     0.0100 48781.7004
    ##      6  3117972.3394             nan     0.0100 38700.8072
    ##      7  3076104.7224             nan     0.0100 40642.4456
    ##      8  3042844.4010             nan     0.0100 23034.0417
    ##      9  3014943.4736             nan     0.0100 21498.2166
    ##     10  2972203.3055             nan     0.0100 42826.1180
    ##     20  2595612.1478             nan     0.0100 27343.3714
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3252692.6714             nan     0.0500 132615.1843
    ##      2  3110010.6607             nan     0.0500 120073.5496
    ##      3  3003251.0111             nan     0.0500 51127.6067
    ##      4  2882922.4116             nan     0.0500 79119.1868
    ##      5  2789662.8129             nan     0.0500 56482.1973
    ##      6  2683038.5754             nan     0.0500 113107.7996
    ##      7  2603276.6138             nan     0.0500 74559.8810
    ##      8  2547428.0898             nan     0.0500 22931.8194
    ##      9  2467407.4153             nan     0.0500 58205.3851
    ##     10  2417607.7324             nan     0.0500 32874.2692
    ##     20  1923625.9923             nan     0.0500 28382.1570
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3169305.0984             nan     0.0500 182970.6432
    ##      2  2987007.3833             nan     0.0500 171700.3289
    ##      3  2787493.9975             nan     0.0500 135900.3618
    ##      4  2640597.6439             nan     0.0500 102313.2539
    ##      5  2469510.9275             nan     0.0500 158452.8742
    ##      6  2320257.4134             nan     0.0500 127552.9709
    ##      7  2170001.3634             nan     0.0500 130770.8312
    ##      8  2030180.3068             nan     0.0500 108185.8383
    ##      9  1920197.1749             nan     0.0500 72253.0422
    ##     10  1800511.0097             nan     0.0500 104859.8563
    ##     20  1050081.6885             nan     0.0500 28049.4826
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3396262.2708             nan     0.0100 28225.1053
    ##      2  3367652.9963             nan     0.0100 28243.2712
    ##      3  3345826.2722             nan     0.0100 25911.5243
    ##      4  3317159.8909             nan     0.0100 18633.4885
    ##      5  3282428.9105             nan     0.0100 20611.3258
    ##      6  3265147.8941             nan     0.0100 16056.0496
    ##      7  3245288.3780             nan     0.0100 16599.5838
    ##      8  3223529.1717             nan     0.0100 24597.7360
    ##      9  3204346.6043             nan     0.0100 14361.5992
    ##     10  3179880.4431             nan     0.0100 19552.4017
    ##     20  2943589.5693             nan     0.0100 13908.3377
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3391229.4902             nan     0.0100 26568.9886
    ##      2  3361648.2147             nan     0.0100 13407.1614
    ##      3  3313335.4602             nan     0.0100 46942.3857
    ##      4  3268213.9067             nan     0.0100 27622.8595
    ##      5  3221161.1594             nan     0.0100 39399.8321
    ##      6  3179196.6478             nan     0.0100 31973.3612
    ##      7  3134468.1397             nan     0.0100 37843.8645
    ##      8  3091546.9736             nan     0.0100 38018.5159
    ##      9  3045051.1829             nan     0.0100 48545.1623
    ##     10  3008585.6625             nan     0.0100 18091.4202
    ##     20  2637757.3875             nan     0.0100 31751.3965
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3273957.9254             nan     0.0500 128569.6031
    ##      2  3139458.6760             nan     0.0500 110439.1253
    ##      3  3015399.9706             nan     0.0500 114934.1941
    ##      4  2925901.7097             nan     0.0500 78472.0272
    ##      5  2805421.5459             nan     0.0500 90116.4866
    ##      6  2722657.6005             nan     0.0500 82445.5813
    ##      7  2648331.1841             nan     0.0500 64032.5660
    ##      8  2570901.7256             nan     0.0500 64621.7673
    ##      9  2483322.2623             nan     0.0500 44765.6288
    ##     10  2409976.1605             nan     0.0500 61145.9754
    ##     20  1868285.6611             nan     0.0500 17921.3906
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3174479.2408             nan     0.0500 244689.1749
    ##      2  2943008.6891             nan     0.0500 198615.3090
    ##      3  2748475.8832             nan     0.0500 191677.0029
    ##      4  2562207.1338             nan     0.0500 130886.6074
    ##      5  2379353.1602             nan     0.0500 130129.5711
    ##      6  2231065.8116             nan     0.0500 131370.9684
    ##      7  2089213.8240             nan     0.0500 101101.0237
    ##      8  1959417.2697             nan     0.0500 85607.3573
    ##      9  1850530.0297             nan     0.0500 76200.0937
    ##     10  1726808.4682             nan     0.0500 93531.4644
    ##     20   990148.2837             nan     0.0500 14406.1214
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3204310.0394             nan     0.0100 18931.9591
    ##      2  3178249.7593             nan     0.0100 18782.2112
    ##      3  3153990.0990             nan     0.0100 21120.3190
    ##      4  3121843.2413             nan     0.0100 26245.3747
    ##      5  3097160.9600             nan     0.0100 20185.6234
    ##      6  3075297.0384             nan     0.0100 20020.7362
    ##      7  3048253.2375             nan     0.0100 18404.1378
    ##      8  3026401.7559             nan     0.0100 20177.2088
    ##      9  3010998.3079             nan     0.0100 8464.8580
    ##     10  2987286.9990             nan     0.0100 19631.1646
    ##     20  2804177.4619             nan     0.0100 19003.4020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3179243.7504             nan     0.0100 42296.3019
    ##      2  3133454.9499             nan     0.0100 33156.1270
    ##      3  3097868.1768             nan     0.0100 25756.4093
    ##      4  3056266.6808             nan     0.0100 40573.7021
    ##      5  3012817.1144             nan     0.0100 42899.0522
    ##      6  2967042.5685             nan     0.0100 30832.3562
    ##      7  2928961.6434             nan     0.0100 25298.0077
    ##      8  2896182.5459             nan     0.0100 23101.9755
    ##      9  2859704.9499             nan     0.0100 36908.8338
    ##     10  2821103.4261             nan     0.0100 33899.2970
    ##     20  2454854.9233             nan     0.0100 29065.5839
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3103630.0397             nan     0.0500 113907.4186
    ##      2  2992964.1737             nan     0.0500 94256.1903
    ##      3  2904246.8104             nan     0.0500 94321.0790
    ##      4  2801557.3733             nan     0.0500 80604.2138
    ##      5  2726897.4340             nan     0.0500 64448.6569
    ##      6  2659425.7878             nan     0.0500 58490.8352
    ##      7  2576330.3339             nan     0.0500 35239.3542
    ##      8  2479967.6380             nan     0.0500 71605.8892
    ##      9  2404516.9751             nan     0.0500 46117.5531
    ##     10  2324546.9190             nan     0.0500 33203.5941
    ##     20  1842884.8392             nan     0.0500 15826.0147
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2996799.2822             nan     0.0500 226717.1680
    ##      2  2783545.0590             nan     0.0500 217229.4018
    ##      3  2615803.8718             nan     0.0500 132516.9113
    ##      4  2460399.7398             nan     0.0500 94293.4557
    ##      5  2310224.7211             nan     0.0500 109441.2392
    ##      6  2191215.0327             nan     0.0500 114490.2741
    ##      7  2071278.6574             nan     0.0500 83913.6306
    ##      8  1925843.7939             nan     0.0500 126944.0774
    ##      9  1806442.5506             nan     0.0500 116273.8943
    ##     10  1697131.1478             nan     0.0500 91982.6689
    ##     20  1016313.3496             nan     0.0500 20347.8752
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3295405.9018             nan     0.0100 25607.3509
    ##      2  3268333.7115             nan     0.0100 22642.4014
    ##      3  3241600.8008             nan     0.0100 23467.8357
    ##      4  3216575.5522             nan     0.0100 24898.3307
    ##      5  3191036.9066             nan     0.0100 20925.7492
    ##      6  3162599.7579             nan     0.0100 20448.3192
    ##      7  3134512.7030             nan     0.0100 12027.4561
    ##      8  3116776.6714             nan     0.0100 14838.4170
    ##      9  3091353.7904             nan     0.0100 16513.8141
    ##     10  3067241.1679             nan     0.0100 18916.7804
    ##     20  2855414.6990             nan     0.0100 9643.4302
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3274293.7488             nan     0.0100 33828.8055
    ##      2  3229485.1631             nan     0.0100 35614.5138
    ##      3  3184053.9822             nan     0.0100 45142.4737
    ##      4  3138601.5423             nan     0.0100 27572.8121
    ##      5  3095633.4590             nan     0.0100 36987.7284
    ##      6  3057285.1024             nan     0.0100 35568.6690
    ##      7  3021717.3083             nan     0.0100 28815.4037
    ##      8  2979776.2956             nan     0.0100 29151.0340
    ##      9  2938271.2257             nan     0.0100 28601.4234
    ##     10  2891767.2538             nan     0.0100 34636.6626
    ##     20  2530129.9990             nan     0.0100 25842.7275
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3176090.4358             nan     0.0500 107897.4780
    ##      2  3082517.0725             nan     0.0500 1449.4798
    ##      3  2981639.0598             nan     0.0500 88712.4886
    ##      4  2901636.2083             nan     0.0500 72709.2292
    ##      5  2803229.6476             nan     0.0500 86790.6541
    ##      6  2720684.8725             nan     0.0500 83017.0321
    ##      7  2631803.4026             nan     0.0500 51496.8649
    ##      8  2569803.2654             nan     0.0500 47263.7377
    ##      9  2474778.3796             nan     0.0500 68241.1952
    ##     10  2420911.6613             nan     0.0500 23126.8885
    ##     20  1899683.3589             nan     0.0500 49131.0229
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3069004.8987             nan     0.0500 187320.2121
    ##      2  2851570.2270             nan     0.0500 152660.6725
    ##      3  2645425.3064             nan     0.0500 180618.1098
    ##      4  2462001.9009             nan     0.0500 117167.4154
    ##      5  2299208.5905             nan     0.0500 137501.3378
    ##      6  2157495.7208             nan     0.0500 116354.8119
    ##      7  2005681.4243             nan     0.0500 86439.4384
    ##      8  1912409.0936             nan     0.0500 57077.0972
    ##      9  1807585.5081             nan     0.0500 105342.6001
    ##     10  1708875.6208             nan     0.0500 47614.6389
    ##     20   991930.4289             nan     0.0500 47966.0631
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3290499.6700             nan     0.0100 23682.0487
    ##      2  3263601.1496             nan     0.0100 24411.7524
    ##      3  3236252.6059             nan     0.0100 22691.0362
    ##      4  3211004.9091             nan     0.0100 21587.2259
    ##      5  3180706.0248             nan     0.0100 17996.5064
    ##      6  3161115.3660             nan     0.0100 20803.1463
    ##      7  3137588.4709             nan     0.0100 20285.2737
    ##      8  3115676.2582             nan     0.0100 18535.4300
    ##      9  3088264.1434             nan     0.0100 14612.9075
    ##     10  3069284.2333             nan     0.0100 14959.4630
    ##     20  2887822.1629             nan     0.0100 16318.5033
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3271448.6189             nan     0.0100 33549.5507
    ##      2  3220819.2960             nan     0.0100 44352.7952
    ##      3  3174309.9290             nan     0.0100 45445.1993
    ##      4  3125022.3695             nan     0.0100 24407.6100
    ##      5  3086797.7124             nan     0.0100 20439.3016
    ##      6  3045629.4208             nan     0.0100 41842.1509
    ##      7  3010967.6676             nan     0.0100 28229.9519
    ##      8  2979271.7405             nan     0.0100 26149.8987
    ##      9  2940777.4995             nan     0.0100 28004.9159
    ##     10  2899334.7554             nan     0.0100 40678.8759
    ##     20  2540614.9731             nan     0.0100 22810.2061
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3156232.3146             nan     0.0500 73756.6850
    ##      2  3043560.9603             nan     0.0500 74752.8146
    ##      3  2947698.7724             nan     0.0500 97078.2185
    ##      4  2829740.5901             nan     0.0500 61836.0447
    ##      5  2759934.3956             nan     0.0500 45695.5527
    ##      6  2718220.0731             nan     0.0500 -15187.7736
    ##      7  2634214.0212             nan     0.0500 63898.8975
    ##      8  2576044.5973             nan     0.0500 56399.4911
    ##      9  2501401.9045             nan     0.0500 53078.4679
    ##     10  2440625.8236             nan     0.0500 54407.3951
    ##     20  1919325.6771             nan     0.0500 24734.6873
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3044483.9060             nan     0.0500 215805.6577
    ##      2  2825090.6108             nan     0.0500 143597.3056
    ##      3  2647027.9826             nan     0.0500 128214.1716
    ##      4  2475056.3426             nan     0.0500 131099.2574
    ##      5  2317261.1516             nan     0.0500 123958.5102
    ##      6  2169657.6423             nan     0.0500 112614.7207
    ##      7  2015096.0304             nan     0.0500 111635.6151
    ##      8  1907773.6180             nan     0.0500 87801.8534
    ##      9  1789130.1721             nan     0.0500 80086.0869
    ##     10  1655663.8834             nan     0.0500 119013.5687
    ##     20   946362.3480             nan     0.0500 26137.7572
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3125224.8094             nan     0.0100 22392.0788
    ##      2  3100279.6473             nan     0.0100 22568.4530
    ##      3  3072632.9032             nan     0.0100 18213.0891
    ##      4  3060962.2417             nan     0.0100 9551.4222
    ##      5  3039067.7398             nan     0.0100 22394.9801
    ##      6  3018958.9739             nan     0.0100 8372.6446
    ##      7  2995247.6178             nan     0.0100 17343.8838
    ##      8  2973092.1495             nan     0.0100 20252.0392
    ##      9  2949546.5997             nan     0.0100 12206.3319
    ##     10  2932844.8649             nan     0.0100 16088.2535
    ##     20  2745970.6029             nan     0.0100 13426.2828
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3111596.1780             nan     0.0100 40760.3054
    ##      2  3068626.8089             nan     0.0100 36603.0829
    ##      3  3024831.3201             nan     0.0100 38946.9022
    ##      4  2984377.9849             nan     0.0100 24666.2170
    ##      5  2944924.6623             nan     0.0100 29154.4348
    ##      6  2903941.7671             nan     0.0100 28228.4757
    ##      7  2866154.2415             nan     0.0100 28591.3080
    ##      8  2830630.6923             nan     0.0100 26577.0919
    ##      9  2785123.6055             nan     0.0100 44551.5020
    ##     10  2748035.0751             nan     0.0100 25983.6045
    ##     20  2402108.0890             nan     0.0100 30586.7349
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3006543.7223             nan     0.0500 113873.8793
    ##      2  2914883.4448             nan     0.0500 76276.6669
    ##      3  2787652.6677             nan     0.0500 90485.8232
    ##      4  2689312.9067             nan     0.0500 73066.3095
    ##      5  2606292.2045             nan     0.0500 69312.4567
    ##      6  2508807.1696             nan     0.0500 55715.4855
    ##      7  2453146.8761             nan     0.0500 34624.7794
    ##      8  2387056.9406             nan     0.0500 70197.8569
    ##      9  2335512.3229             nan     0.0500 18069.3480
    ##     10  2283683.5650             nan     0.0500 41889.2596
    ##     20  1799257.1799             nan     0.0500 14435.5615
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  2944674.8037             nan     0.0500 259459.2780
    ##      2  2754299.1789             nan     0.0500 167901.9983
    ##      3  2578172.0084             nan     0.0500 122994.3193
    ##      4  2410775.9878             nan     0.0500 133021.2490
    ##      5  2239819.5521             nan     0.0500 172399.7378
    ##      6  2103418.5300             nan     0.0500 95203.9583
    ##      7  1965754.3644             nan     0.0500 80730.3365
    ##      8  1827665.6063             nan     0.0500 69353.3486
    ##      9  1706473.3494             nan     0.0500 51625.0399
    ##     10  1597330.7158             nan     0.0500 101801.8203
    ##     20   945270.7951             nan     0.0500 21781.4558
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3421904.0081             nan     0.0100 21435.3643
    ##      2  3393283.2995             nan     0.0100 19302.3875
    ##      3  3369794.3786             nan     0.0100 9692.1771
    ##      4  3334783.9575             nan     0.0100 20045.2189
    ##      5  3305664.1694             nan     0.0100 23691.4325
    ##      6  3284202.7769             nan     0.0100 21678.8648
    ##      7  3278950.3825             nan     0.0100 -3428.5542
    ##      8  3253966.8557             nan     0.0100 19329.1157
    ##      9  3224132.0007             nan     0.0100 13945.1361
    ##     10  3196685.7003             nan     0.0100 20289.7073
    ##     20  2981777.6472             nan     0.0100 11816.9524
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3412782.1571             nan     0.0100 40321.7743
    ##      2  3362553.6472             nan     0.0100 32028.5232
    ##      3  3319723.4991             nan     0.0100 32880.3298
    ##      4  3272433.1607             nan     0.0100 40878.9947
    ##      5  3226632.7624             nan     0.0100 35821.1857
    ##      6  3174035.5804             nan     0.0100 38499.5040
    ##      7  3132146.7485             nan     0.0100 22821.5246
    ##      8  3084911.4778             nan     0.0100 46041.8476
    ##      9  3043947.8873             nan     0.0100 47652.2201
    ##     10  3010422.0427             nan     0.0100 31532.3467
    ##     20  2626169.3873             nan     0.0100 34749.5473
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3299375.2162             nan     0.0500 126991.8470
    ##      2  3185990.4457             nan     0.0500 113593.9472
    ##      3  3051112.1514             nan     0.0500 54392.9539
    ##      4  2942920.4248             nan     0.0500 94754.3054
    ##      5  2838189.6966             nan     0.0500 90056.2331
    ##      6  2711436.6236             nan     0.0500 63372.2049
    ##      7  2599180.6388             nan     0.0500 50069.5555
    ##      8  2506149.2450             nan     0.0500 78286.6740
    ##      9  2415390.4128             nan     0.0500 46143.9868
    ##     10  2364729.0018             nan     0.0500 28631.5981
    ##     20  1807266.8788             nan     0.0500 39286.2094
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3221270.8195             nan     0.0500 177179.8605
    ##      2  2989980.3702             nan     0.0500 207012.1958
    ##      3  2781549.6978             nan     0.0500 133231.2108
    ##      4  2608601.3897             nan     0.0500 150446.8316
    ##      5  2434783.3456             nan     0.0500 159422.7059
    ##      6  2283776.6815             nan     0.0500 140749.5641
    ##      7  2143330.6190             nan     0.0500 129514.5659
    ##      8  1991652.9629             nan     0.0500 144455.5700
    ##      9  1846136.0003             nan     0.0500 90093.1213
    ##     10  1738094.2516             nan     0.0500 49186.3646
    ##     20  1004500.4214             nan     0.0500 29855.8077
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3085231.0676             nan     0.0500 210079.3966
    ##      2  2893572.0631             nan     0.0500 184061.4150
    ##      3  2713405.9476             nan     0.0500 172387.0518
    ##      4  2525512.9132             nan     0.0500 170634.9348
    ##      5  2371140.3488             nan     0.0500 105143.1427
    ##      6  2203105.3894             nan     0.0500 158809.4714
    ##      7  2046300.5552             nan     0.0500 80185.3873
    ##      8  1913158.9828             nan     0.0500 86282.3252
    ##      9  1801177.3526             nan     0.0500 70345.2063
    ##     10  1692606.0607             nan     0.0500 83210.5559
    ##     20   952308.2966             nan     0.0500 30283.2626

``` r
# Display information from the boost fit
fitBoost$results
```

    ##   shrinkage interaction.depth n.minobsinnode
    ## 1      0.01                 1              3
    ## 5      0.05                 1              3
    ## 3      0.01                 5              3
    ## 7      0.05                 5              3
    ## 2      0.01                 1              3
    ## 6      0.05                 1              3
    ## 4      0.01                 5              3
    ## 8      0.05                 5              3
    ##   n.trees     RMSE  Rsquared       MAE   RMSESD
    ## 1      10 1734.925 0.4480198 1434.8610 442.3957
    ## 5      10 1536.637 0.5824130 1268.0836 450.0916
    ## 3      10 1690.517 0.6983663 1402.1007 436.6668
    ## 7      10 1371.919 0.6945792 1113.6907 445.8834
    ## 2      20 1682.075 0.4823198 1391.7032 443.0131
    ## 6      20 1394.853 0.6201307 1140.2060 469.5742
    ## 4      20 1589.005 0.7156397 1317.4303 436.3682
    ## 8      20 1133.081 0.7280664  891.9338 454.2683
    ##   RsquaredSD    MAESD
    ## 1  0.1696834 382.3785
    ## 5  0.2208120 390.9412
    ## 3  0.2535106 370.5966
    ## 7  0.2644414 365.2358
    ## 2  0.1879347 382.3446
    ## 6  0.2333821 380.5306
    ## 4  0.2529020 367.8450
    ## 8  0.2427635 353.7345

Now, we make predictions on the test data sets using the best model
fits. Then we compare RMSE to determine the best model.

``` r
predTree <- predict(fitTree, newdata = select(dayTest, -cnt))
postResample(predTree, dayTest$cnt)
```

    ##        RMSE    Rsquared         MAE 
    ## 977.2493564   0.6922567 753.2120197

``` r
boostPred <- predict(fitBoost, newdata = select(dayTest, -cnt))
postResample(boostPred, dayTest$cnt)
```

    ##        RMSE    Rsquared         MAE 
    ## 805.8514004   0.8896831 664.9933563

When we compare the two models, the boosted tree model has lower RMSE
values when applied on the test dataset. Hence, the boosted tree model
is our final model and best model for interpreting the bike rental count
on a daily basis.
