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

    ## 'data.frame':    104 obs. of  14 variables:
    ##  $ instant   : int  6 13 20 27 34 41 48 55 62 69 ...
    ##  $ dteday    : chr  "2011-01-06" "2011-01-13" "2011-01-20" "2011-01-27" ...
    ##  $ season    : Factor w/ 4 levels "winter","spring",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ yr        : Factor w/ 2 levels "2011","2012": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ mnth      : Factor w/ 12 levels "1","2","3","4",..: 1 1 1 1 2 2 2 2 3 3 ...
    ##  $ holiday   : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weekday   : Factor w/ 7 levels "Sunday","Monday",..: 5 5 5 5 5 5 5 5 5 5 ...
    ##  $ workingday: Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ weathersit: Factor w/ 3 levels "1","2","3": 1 1 2 1 1 1 1 2 1 3 ...
    ##  $ temp      : num  0.204 0.165 0.262 0.195 0.187 ...
    ##  $ atemp     : num  0.233 0.151 0.255 0.22 0.178 ...
    ##  $ hum       : num  0.518 0.47 0.538 0.688 0.438 ...
    ##  $ windspeed : num  0.0896 0.301 0.1959 0.1138 0.2778 ...
    ##  $ cnt       : int  1606 1406 1927 431 1550 1538 2475 1807 1685 623 ...

### Univariate Analysis

The `cnt` is the response variable, so we’ll use a histogram to get a
visual understanding of the variable.

``` r
ggplot(day, aes(x = cnt)) + theme_bw() + geom_histogram(aes(y =..density..), color = "black", fill = "white", binwidth = 1000) + geom_density(alpha = 0.2, fill = "blue") + labs(title = "Count Density", x = "Count", y = "Density")
```

![](Thursday_files/figure-gfm/cnt-1.png)<!-- -->

``` r
summary(day$cnt)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     431    3271    4721    4667    6286    7804

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

![](Thursday_files/figure-gfm/EDA_num-1.png)<!-- -->

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

![](Thursday_files/figure-gfm/EDA_cat-1.png)<!-- -->

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

![](Thursday_files/figure-gfm/bi_var_num-1.png)<!-- -->

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

![](Thursday_files/figure-gfm/bivar_cat-1.png)<!-- -->

``` r
gridExtra::grid.arrange(g4, g6, g7, nrow = 2)
```

![](Thursday_files/figure-gfm/bivar_cat-2.png)<!-- --> Observations:  
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
    ## temp       1.00  1.00  0.15     -0.11  0.60
    ## atemp      1.00  1.00  0.15     -0.12  0.61
    ## hum        0.15  0.15  1.00     -0.31  0.00
    ## windspeed -0.11 -0.12 -0.31      1.00 -0.18
    ## cnt        0.60  0.61  0.00     -0.18  1.00

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

    ## Warning in nominalTrainWorkflow(x = x, y =
    ## y, wts = weights, info = trainInfo, : There
    ## were missing values in resampled performance
    ## measures.

``` r
# Display information from the tree fit
fitTree$results
```

    ##          cp     RMSE Rsquared      MAE   RMSESD
    ## 1 0.1623048 1094.253      NaN 1094.253 654.7690
    ## 2 0.2128773 1338.501      NaN 1338.501 800.1816
    ## 3 0.4540853 2025.910      NaN 2025.910 917.6932
    ##   RsquaredSD    MAESD
    ## 1         NA 654.7690
    ## 2         NA 800.1816
    ## 3         NA 917.6932

``` r
# Build a boosted tree model using cv
fitBoost <- train(cnt~., data = dayTrain, method = "gbm", 
              preProcess = c("center", "scale"), 
              trControl = trainControl(method = "cv", number = 10), 
              tuneGrid = NULL)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3517945.5172             nan     0.1000 277459.9978
    ##      2  3264990.5990             nan     0.1000 298067.8291
    ##      3  2987491.2562             nan     0.1000 226035.6902
    ##      4  2678373.7730             nan     0.1000 209470.6508
    ##      5  2432960.1480             nan     0.1000 195722.5225
    ##      6  2230315.4747             nan     0.1000 186121.3480
    ##      7  2032687.8358             nan     0.1000 119106.9190
    ##      8  1891492.0731             nan     0.1000 120858.8699
    ##      9  1725522.7243             nan     0.1000 118863.0261
    ##     10  1608660.4677             nan     0.1000 86732.3048
    ##     20   900230.6926             nan     0.1000 24087.2248
    ##     40   549101.6965             nan     0.1000 -10434.7282
    ##     60   464831.3197             nan     0.1000 -2324.4034
    ##     80   436397.6581             nan     0.1000 -2860.6127
    ##    100   402031.1533             nan     0.1000 -5630.4229
    ##    120   364484.1351             nan     0.1000 -2701.6785
    ##    140   344244.4163             nan     0.1000 -2782.4477
    ##    150   336729.6685             nan     0.1000 -6316.6447
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3483283.1573             nan     0.1000 296355.5884
    ##      2  3004201.7201             nan     0.1000 393740.2054
    ##      3  2697569.9135             nan     0.1000 236630.2700
    ##      4  2499902.4683             nan     0.1000 188557.8791
    ##      5  2263097.0178             nan     0.1000 184329.9475
    ##      6  2139607.6179             nan     0.1000 99683.7149
    ##      7  2007541.7971             nan     0.1000 128765.6883
    ##      8  1859356.4668             nan     0.1000 122808.4493
    ##      9  1670364.0594             nan     0.1000 184789.5960
    ##     10  1533231.5492             nan     0.1000 80043.8916
    ##     20   867902.4294             nan     0.1000 29861.3882
    ##     40   530911.4698             nan     0.1000 15441.5063
    ##     60   436644.6259             nan     0.1000 -7409.4660
    ##     80   387239.3820             nan     0.1000 -5128.7824
    ##    100   348290.8872             nan     0.1000 -8955.8016
    ##    120   317459.2112             nan     0.1000 1236.4558
    ##    140   293982.4253             nan     0.1000 -2572.4339
    ##    150   275340.1227             nan     0.1000 -4121.2669
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3524150.1813             nan     0.1000 317130.2638
    ##      2  3160420.3897             nan     0.1000 287535.2708
    ##      3  2889264.6494             nan     0.1000 185981.3070
    ##      4  2590940.7368             nan     0.1000 251958.4368
    ##      5  2356871.5113             nan     0.1000 220839.5988
    ##      6  2161083.9832             nan     0.1000 224315.2252
    ##      7  1980824.7126             nan     0.1000 147230.0149
    ##      8  1819296.0515             nan     0.1000 105349.0551
    ##      9  1721464.6474             nan     0.1000 91028.2108
    ##     10  1654387.9227             nan     0.1000 7629.2365
    ##     20   839122.5590             nan     0.1000 31051.3374
    ##     40   499412.9861             nan     0.1000 -2035.9316
    ##     60   409293.0640             nan     0.1000 -3325.4709
    ##     80   371668.7068             nan     0.1000 -5693.2473
    ##    100   339106.8083             nan     0.1000 -1863.3142
    ##    120   320134.0361             nan     0.1000 -5396.3516
    ##    140   292501.4178             nan     0.1000 -5678.2503
    ##    150   282974.0261             nan     0.1000 -7826.1803
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3503468.8122             nan     0.1000 298817.1270
    ##      2  3228894.4272             nan     0.1000 250423.1330
    ##      3  2890164.2044             nan     0.1000 253727.2101
    ##      4  2681862.1828             nan     0.1000 211502.2341
    ##      5  2471447.7964             nan     0.1000 175402.0087
    ##      6  2261791.4297             nan     0.1000 100123.9465
    ##      7  2110384.3617             nan     0.1000 130832.3337
    ##      8  1948443.9194             nan     0.1000 141557.4874
    ##      9  1837207.3272             nan     0.1000 114053.2815
    ##     10  1733862.0372             nan     0.1000 89803.3187
    ##     20  1014021.3304             nan     0.1000 13022.1167
    ##     40   639982.9985             nan     0.1000 6372.7338
    ##     60   529430.1791             nan     0.1000 -138.8838
    ##     80   481182.3128             nan     0.1000 -1097.8594
    ##    100   442162.2112             nan     0.1000 -7615.0290
    ##    120   418020.5122             nan     0.1000 -5419.4676
    ##    140   399568.2794             nan     0.1000 -4975.3764
    ##    150   390582.2072             nan     0.1000 -3830.3651
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3498110.5453             nan     0.1000 313911.2437
    ##      2  3210647.8961             nan     0.1000 278313.7166
    ##      3  2980046.9633             nan     0.1000 204401.3565
    ##      4  2749164.9324             nan     0.1000 244028.1342
    ##      5  2534477.0175             nan     0.1000 189439.5886
    ##      6  2329638.0131             nan     0.1000 185731.6155
    ##      7  2161644.9094             nan     0.1000 174492.6950
    ##      8  1980309.3302             nan     0.1000 135980.7634
    ##      9  1817287.6353             nan     0.1000 81245.6859
    ##     10  1618900.9812             nan     0.1000 153675.0602
    ##     20   922992.7245             nan     0.1000 -2310.2349
    ##     40   604651.9845             nan     0.1000 -11735.3717
    ##     60   497705.3076             nan     0.1000 -8049.7113
    ##     80   444644.5415             nan     0.1000 -2236.2441
    ##    100   399249.2596             nan     0.1000 -13776.3767
    ##    120   372649.5249             nan     0.1000 -8622.1662
    ##    140   335669.3082             nan     0.1000 -7060.4751
    ##    150   325321.6433             nan     0.1000 -804.8944
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3565819.5438             nan     0.1000 323553.2348
    ##      2  3265258.3530             nan     0.1000 336737.5131
    ##      3  2997882.5063             nan     0.1000 259336.5647
    ##      4  2774348.8914             nan     0.1000 248312.2199
    ##      5  2415757.4035             nan     0.1000 278106.0583
    ##      6  2238477.0313             nan     0.1000 177319.4290
    ##      7  2089997.0428             nan     0.1000 155909.6754
    ##      8  1904191.3420             nan     0.1000 117313.5189
    ##      9  1736134.3935             nan     0.1000 172977.7984
    ##     10  1610604.5826             nan     0.1000 86766.7470
    ##     20   923111.7782             nan     0.1000  583.2284
    ##     40   611013.8192             nan     0.1000 2045.6988
    ##     60   529195.4030             nan     0.1000 -3833.5957
    ##     80   475479.5163             nan     0.1000 -16608.2442
    ##    100   440004.3816             nan     0.1000 -5960.2238
    ##    120   402085.4914             nan     0.1000 -3254.8243
    ##    140   371261.6178             nan     0.1000 -6155.1466
    ##    150   354191.0555             nan     0.1000 -6292.9776
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3513862.8881             nan     0.1000 294263.7960
    ##      2  3201350.2358             nan     0.1000 316795.5810
    ##      3  2940884.6195             nan     0.1000 253353.3570
    ##      4  2687553.4297             nan     0.1000 205569.3036
    ##      5  2422774.5364             nan     0.1000 194392.8713
    ##      6  2286262.6868             nan     0.1000 150585.6199
    ##      7  2132975.0134             nan     0.1000 164133.7915
    ##      8  1986614.1513             nan     0.1000 131744.0181
    ##      9  1896967.1249             nan     0.1000 122795.5941
    ##     10  1778890.0566             nan     0.1000 10965.4241
    ##     20  1119812.5959             nan     0.1000 -19949.5857
    ##     40   674820.6614             nan     0.1000 -17279.5088
    ##     60   552209.2873             nan     0.1000 -4702.4954
    ##     80   487216.3630             nan     0.1000 -7042.1099
    ##    100   467438.4878             nan     0.1000 -2491.4195
    ##    120   431752.9311             nan     0.1000 -5900.8299
    ##    140   399195.0621             nan     0.1000 -2424.8262
    ##    150   396929.7730             nan     0.1000 -5196.9480
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3480255.4590             nan     0.1000 322677.7018
    ##      2  3141983.1602             nan     0.1000 303224.4983
    ##      3  2920305.9612             nan     0.1000 235439.5061
    ##      4  2611812.4876             nan     0.1000 200299.0678
    ##      5  2469462.8737             nan     0.1000 161831.7326
    ##      6  2276652.8672             nan     0.1000 169112.2152
    ##      7  2176688.7514             nan     0.1000 39248.2261
    ##      8  1924893.2168             nan     0.1000 232651.7825
    ##      9  1802335.5963             nan     0.1000 104883.8759
    ##     10  1660198.0616             nan     0.1000 93985.8165
    ##     20   899317.0363             nan     0.1000 37277.4598
    ##     40   597123.6598             nan     0.1000 -13831.6742
    ##     60   490073.8731             nan     0.1000 -9180.4123
    ##     80   436846.2307             nan     0.1000 -5239.0575
    ##    100   387172.3097             nan     0.1000 -6485.7915
    ##    120   355978.5075             nan     0.1000 -4902.4449
    ##    140   319488.5422             nan     0.1000 -3886.0838
    ##    150   312987.9690             nan     0.1000 -10397.5531
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3482848.0589             nan     0.1000 288350.6257
    ##      2  3086761.7123             nan     0.1000 339493.5844
    ##      3  2892423.5419             nan     0.1000 238614.6290
    ##      4  2563451.0070             nan     0.1000 335968.0672
    ##      5  2282871.8205             nan     0.1000 280421.4606
    ##      6  2092556.9171             nan     0.1000 167166.8128
    ##      7  1945154.0346             nan     0.1000 132965.0046
    ##      8  1816202.4946             nan     0.1000 82787.6112
    ##      9  1664347.2283             nan     0.1000 129715.7032
    ##     10  1534512.7370             nan     0.1000 89649.5930
    ##     20   903394.0007             nan     0.1000 34330.0096
    ##     40   587553.5467             nan     0.1000 -10715.9963
    ##     60   509398.7467             nan     0.1000 -13020.5341
    ##     80   452361.9696             nan     0.1000 -4559.8345
    ##    100   395198.5624             nan     0.1000 -5878.7514
    ##    120   351160.6578             nan     0.1000  749.0115
    ##    140   325274.2497             nan     0.1000 -7859.0698
    ##    150   315814.9812             nan     0.1000 -3143.1982
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3311182.5302             nan     0.1000 261674.8090
    ##      2  3005207.2161             nan     0.1000 278436.6905
    ##      3  2764958.6241             nan     0.1000 188121.5223
    ##      4  2562646.8755             nan     0.1000 224995.6587
    ##      5  2367084.4131             nan     0.1000 192756.4303
    ##      6  2165432.5562             nan     0.1000 124703.7881
    ##      7  1988219.5848             nan     0.1000 148677.4137
    ##      8  1840421.1891             nan     0.1000 139628.2592
    ##      9  1700802.3151             nan     0.1000 110123.0041
    ##     10  1606375.1317             nan     0.1000 74991.8707
    ##     20   958192.4870             nan     0.1000 39620.3874
    ##     40   577817.6013             nan     0.1000 3811.6250
    ##     60   484355.9907             nan     0.1000  393.2570
    ##     80   445492.1261             nan     0.1000 -8179.3693
    ##    100   419859.9575             nan     0.1000 -4599.9055
    ##    120   401503.4889             nan     0.1000 -4882.4531
    ##    140   377735.0303             nan     0.1000 -5578.7401
    ##    150   373005.1705             nan     0.1000 -512.7109
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3199974.6829             nan     0.1000 229763.0603
    ##      2  2892442.7298             nan     0.1000 206558.3693
    ##      3  2600735.3498             nan     0.1000 217901.8964
    ##      4  2413852.0736             nan     0.1000 114044.1407
    ##      5  2146906.8808             nan     0.1000 230439.8570
    ##      6  1972761.9713             nan     0.1000 182949.0026
    ##      7  1884561.7528             nan     0.1000 81151.7533
    ##      8  1758242.0684             nan     0.1000 72400.3788
    ##      9  1606282.7209             nan     0.1000 146057.9990
    ##     10  1491197.8455             nan     0.1000 119686.9570
    ##     20   838392.4188             nan     0.1000 -15033.6591
    ##     40   541037.8026             nan     0.1000 -207.3156
    ##     60   459195.9910             nan     0.1000 -5530.4119
    ##     80   410142.1910             nan     0.1000 -15450.3613
    ##    100   363350.4379             nan     0.1000 -6322.5702
    ##    120   331041.7184             nan     0.1000 -7661.7699
    ##    140   301721.0147             nan     0.1000 -4475.6397
    ##    150   297909.8259             nan     0.1000 -5166.5822
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3292333.6196             nan     0.1000 303437.2574
    ##      2  2994221.3764             nan     0.1000 302785.3269
    ##      3  2632552.8898             nan     0.1000 386695.0407
    ##      4  2411747.2988             nan     0.1000 233284.9611
    ##      5  2233316.4017             nan     0.1000 180691.5784
    ##      6  2095611.6070             nan     0.1000 129173.9035
    ##      7  1936773.7006             nan     0.1000 87809.7023
    ##      8  1760282.3202             nan     0.1000 142866.8885
    ##      9  1656800.1537             nan     0.1000 107633.3150
    ##     10  1555508.3718             nan     0.1000 96152.5899
    ##     20   843776.5246             nan     0.1000 8695.2734
    ##     40   535784.3265             nan     0.1000 1193.2949
    ##     60   456551.0022             nan     0.1000 -5318.6100
    ##     80   425465.4220             nan     0.1000 -597.9134
    ##    100   378025.7104             nan     0.1000 -4850.2568
    ##    120   349829.5889             nan     0.1000  -93.5541
    ##    140   329849.8046             nan     0.1000 -6675.3856
    ##    150   311402.2790             nan     0.1000 -5693.6220
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3531450.8748             nan     0.1000 323746.1113
    ##      2  3274232.2846             nan     0.1000 221574.7768
    ##      3  2990158.2716             nan     0.1000 292892.0198
    ##      4  2736333.5667             nan     0.1000 201776.7418
    ##      5  2546765.0039             nan     0.1000 205438.8899
    ##      6  2405117.8538             nan     0.1000 114440.6553
    ##      7  2207011.4545             nan     0.1000 165684.5074
    ##      8  2063665.8707             nan     0.1000 163519.3783
    ##      9  1909642.8776             nan     0.1000 137832.2408
    ##     10  1819944.6278             nan     0.1000 52105.5157
    ##     20  1040711.2674             nan     0.1000 30223.1653
    ##     40   661997.7911             nan     0.1000 -2619.8519
    ##     60   581474.2426             nan     0.1000 -2331.6626
    ##     80   529081.0736             nan     0.1000 -4547.1707
    ##    100   496541.2255             nan     0.1000 -3855.9246
    ##    120   468452.3990             nan     0.1000 -12145.8381
    ##    140   447077.5239             nan     0.1000  922.8731
    ##    150   436419.3663             nan     0.1000 -11837.1730
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3610247.5386             nan     0.1000 125671.3738
    ##      2  3255110.1003             nan     0.1000 314089.9947
    ##      3  3074850.5003             nan     0.1000 90474.4834
    ##      4  2798275.4708             nan     0.1000 222436.2864
    ##      5  2448402.6411             nan     0.1000 324600.7930
    ##      6  2207479.8804             nan     0.1000 273602.6548
    ##      7  2031302.4759             nan     0.1000 157898.9894
    ##      8  1893298.9266             nan     0.1000 130884.2969
    ##      9  1765523.2903             nan     0.1000 112346.7874
    ##     10  1622763.5891             nan     0.1000 87064.7155
    ##     20   896545.2016             nan     0.1000 20660.0633
    ##     40   598394.6158             nan     0.1000 -471.3023
    ##     60   531165.4865             nan     0.1000 -6741.2636
    ##     80   472612.3271             nan     0.1000 -8732.7279
    ##    100   419918.9477             nan     0.1000 -8344.7664
    ##    120   394105.1802             nan     0.1000 -10892.0401
    ##    140   357953.8165             nan     0.1000 -12318.6806
    ##    150   345795.0423             nan     0.1000 -13950.5108
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3541775.7513             nan     0.1000 264607.9745
    ##      2  3268822.2948             nan     0.1000 313372.3451
    ##      3  2976599.5275             nan     0.1000 269030.3143
    ##      4  2773967.7468             nan     0.1000 216757.2880
    ##      5  2435842.3233             nan     0.1000 316995.4705
    ##      6  2199910.2900             nan     0.1000 179572.3937
    ##      7  2076887.0398             nan     0.1000 119557.6025
    ##      8  1870801.6500             nan     0.1000 186026.2336
    ##      9  1712920.2025             nan     0.1000 159780.7026
    ##     10  1531419.2110             nan     0.1000 90862.4410
    ##     20   896453.5454             nan     0.1000 5715.5489
    ##     40   588829.5098             nan     0.1000 -3112.1913
    ##     60   510802.1619             nan     0.1000 -2356.4435
    ##     80   441060.9354             nan     0.1000 -5336.6011
    ##    100   413099.8063             nan     0.1000 -4504.1314
    ##    120   375029.4456             nan     0.1000 -8620.5111
    ##    140   351342.3837             nan     0.1000 -10306.0436
    ##    150   340825.3922             nan     0.1000 -2062.9629
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3428523.4028             nan     0.1000 388390.1289
    ##      2  3168157.4403             nan     0.1000 287604.7781
    ##      3  2899141.4611             nan     0.1000 299872.7702
    ##      4  2647244.3830             nan     0.1000 260645.6472
    ##      5  2419313.8117             nan     0.1000 241600.8464
    ##      6  2231128.9760             nan     0.1000 160489.5631
    ##      7  2015688.1546             nan     0.1000 190296.5935
    ##      8  1848459.3880             nan     0.1000 159598.3019
    ##      9  1717922.3302             nan     0.1000 140717.3547
    ##     10  1592501.4685             nan     0.1000 97910.9268
    ##     20   881088.9837             nan     0.1000 -4005.2584
    ##     40   506469.7488             nan     0.1000 -559.7558
    ##     60   436695.8834             nan     0.1000 -3208.2773
    ##     80   385415.3542             nan     0.1000 -2108.9056
    ##    100   361443.5205             nan     0.1000 -6329.3840
    ##    120   340568.9123             nan     0.1000 -1388.0578
    ##    140   321735.2145             nan     0.1000 -5353.7204
    ##    150   314992.1449             nan     0.1000 -4209.0725
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3384199.3229             nan     0.1000 336932.5695
    ##      2  3091741.4727             nan     0.1000 344849.0751
    ##      3  2769811.7487             nan     0.1000 239662.5379
    ##      4  2471243.3715             nan     0.1000 280711.5182
    ##      5  2256265.1729             nan     0.1000 228181.8914
    ##      6  2069176.8721             nan     0.1000 183864.6037
    ##      7  1786748.3510             nan     0.1000 217328.8448
    ##      8  1654698.9483             nan     0.1000 120588.0090
    ##      9  1468792.4093             nan     0.1000 128762.1595
    ##     10  1403200.1658             nan     0.1000 81948.8480
    ##     20   749949.5347             nan     0.1000 2470.5726
    ##     40   460048.6806             nan     0.1000  493.8278
    ##     60   392253.4270             nan     0.1000 -7153.6994
    ##     80   348663.5998             nan     0.1000 -12028.9809
    ##    100   325408.6507             nan     0.1000 -3747.1458
    ##    120   294614.6466             nan     0.1000 -2487.5989
    ##    140   272593.7373             nan     0.1000 -6370.0226
    ##    150   264075.1455             nan     0.1000 -2571.1105
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3545571.3745             nan     0.1000 348650.7711
    ##      2  3120148.5916             nan     0.1000 310842.7046
    ##      3  2852876.8009             nan     0.1000 288115.2802
    ##      4  2532685.8804             nan     0.1000 173238.3856
    ##      5  2278133.8446             nan     0.1000 241866.4951
    ##      6  2131219.7161             nan     0.1000 189791.6049
    ##      7  1943761.7185             nan     0.1000 160314.1426
    ##      8  1793666.4141             nan     0.1000 127135.3655
    ##      9  1554107.2889             nan     0.1000 181546.8535
    ##     10  1431780.1611             nan     0.1000 109767.8184
    ##     20   762826.3748             nan     0.1000 5220.3741
    ##     40   497924.0096             nan     0.1000 -10324.5316
    ##     60   425972.3762             nan     0.1000 -2859.2061
    ##     80   393884.2105             nan     0.1000 -5247.7991
    ##    100   342789.8635             nan     0.1000 -689.1676
    ##    120   315590.6708             nan     0.1000 -5595.8669
    ##    140   284941.5544             nan     0.1000 -6008.0605
    ##    150   277060.7589             nan     0.1000 -7864.7344
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3424406.7826             nan     0.1000 333704.8169
    ##      2  3153653.3449             nan     0.1000 263279.3622
    ##      3  2843205.3336             nan     0.1000 230585.3286
    ##      4  2631383.8640             nan     0.1000 217538.5342
    ##      5  2466108.4286             nan     0.1000 43711.7121
    ##      6  2263833.1855             nan     0.1000 147670.7770
    ##      7  2087504.1833             nan     0.1000 151073.6573
    ##      8  1902128.8614             nan     0.1000 154659.4542
    ##      9  1738984.5878             nan     0.1000 80530.3747
    ##     10  1603221.5894             nan     0.1000 98477.3127
    ##     20   978760.2140             nan     0.1000 4801.0114
    ##     40   633189.8072             nan     0.1000 -6267.3505
    ##     60   545491.0694             nan     0.1000 -2483.3248
    ##     80   493342.4278             nan     0.1000 -6261.0077
    ##    100   468481.4905             nan     0.1000 -6976.6166
    ##    120   448437.8197             nan     0.1000 -9184.4792
    ##    140   426069.7530             nan     0.1000 -3622.2780
    ##    150   421168.2327             nan     0.1000 -7812.5502
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3406368.0640             nan     0.1000 314862.8538
    ##      2  3084210.2267             nan     0.1000 241900.6934
    ##      3  2693609.7648             nan     0.1000 242412.5210
    ##      4  2487503.5828             nan     0.1000 203767.4239
    ##      5  2280188.4386             nan     0.1000 174135.0117
    ##      6  2103503.7396             nan     0.1000 110733.6747
    ##      7  1949562.0066             nan     0.1000 131127.2645
    ##      8  1862541.2655             nan     0.1000 90576.0395
    ##      9  1709247.2276             nan     0.1000 159257.2546
    ##     10  1569672.4459             nan     0.1000 125449.0310
    ##     20   887588.5650             nan     0.1000 3484.7020
    ##     40   569612.1145             nan     0.1000 -1888.2300
    ##     60   485608.6993             nan     0.1000 -3610.1341
    ##     80   450875.2073             nan     0.1000 -2396.2566
    ##    100   410849.2445             nan     0.1000 -4001.0091
    ##    120   371081.9519             nan     0.1000 -8522.4124
    ##    140   342435.9706             nan     0.1000 -5501.5273
    ##    150   331679.1808             nan     0.1000 -5870.1598
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3310874.3936             nan     0.1000 331407.6729
    ##      2  3042826.5186             nan     0.1000 258914.0342
    ##      3  2767311.4254             nan     0.1000 188366.1819
    ##      4  2549860.3234             nan     0.1000 203260.8346
    ##      5  2217046.4096             nan     0.1000 324988.6978
    ##      6  1954624.8235             nan     0.1000 254651.4416
    ##      7  1829271.8299             nan     0.1000 134680.0252
    ##      8  1709281.8960             nan     0.1000 95906.7453
    ##      9  1577927.7318             nan     0.1000 132178.9581
    ##     10  1480216.7016             nan     0.1000 89904.4908
    ##     20   849249.0863             nan     0.1000 -211.4326
    ##     40   576433.0119             nan     0.1000 -2021.2748
    ##     60   493943.6759             nan     0.1000 -14646.5631
    ##     80   460085.9456             nan     0.1000 -4286.1879
    ##    100   419348.1256             nan     0.1000 -13678.3272
    ##    120   390963.3446             nan     0.1000 -4784.0651
    ##    140   364687.3541             nan     0.1000 -10080.5540
    ##    150   355358.1389             nan     0.1000 -2790.4802
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3519431.7506             nan     0.1000 322573.9811
    ##      2  3167985.6652             nan     0.1000 256033.6629
    ##      3  2923025.9326             nan     0.1000 178621.3425
    ##      4  2614966.5999             nan     0.1000 307315.5721
    ##      5  2364093.5661             nan     0.1000 152197.8211
    ##      6  2179429.3171             nan     0.1000 194222.7661
    ##      7  2005444.5086             nan     0.1000 149587.4492
    ##      8  1889768.2305             nan     0.1000 102128.9269
    ##      9  1794331.7744             nan     0.1000 87203.3213
    ##     10  1685340.1175             nan     0.1000 81846.7651
    ##     20   912084.4601             nan     0.1000 48140.7384
    ##     40   625649.6497             nan     0.1000 -3527.1394
    ##     60   531509.3068             nan     0.1000 -6927.3027
    ##     80   484951.4877             nan     0.1000 -3315.7069
    ##    100   448809.3224             nan     0.1000 -7044.8567
    ##    120   426402.8733             nan     0.1000 -9076.2549
    ##    140   402393.3443             nan     0.1000 -13322.3406
    ##    150   394225.5570             nan     0.1000 -4867.6979
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3516407.6074             nan     0.1000 316358.6364
    ##      2  3052411.6648             nan     0.1000 384608.0610
    ##      3  2656141.2826             nan     0.1000 314041.0760
    ##      4  2331732.8885             nan     0.1000 286117.7011
    ##      5  2171297.2661             nan     0.1000 122261.0014
    ##      6  1923312.2269             nan     0.1000 242752.1865
    ##      7  1778709.3661             nan     0.1000 136697.2826
    ##      8  1594098.4377             nan     0.1000 206901.5340
    ##      9  1479831.5445             nan     0.1000 84510.9682
    ##     10  1404612.2570             nan     0.1000 89996.9363
    ##     20   877601.4049             nan     0.1000 36616.8573
    ##     40   561229.7198             nan     0.1000 -8710.5814
    ##     60   478580.9473             nan     0.1000 -12366.5926
    ##     80   444831.6625             nan     0.1000 -21563.9201
    ##    100   407353.0649             nan     0.1000 -7080.8711
    ##    120   384093.8233             nan     0.1000 -3955.5096
    ##    140   367251.0478             nan     0.1000 -14660.0605
    ##    150   348059.3877             nan     0.1000 -9062.3581
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3514411.3653             nan     0.1000 317055.4862
    ##      2  3236498.9238             nan     0.1000 272237.0353
    ##      3  2840237.9959             nan     0.1000 328295.8853
    ##      4  2543478.0636             nan     0.1000 241619.9396
    ##      5  2279620.8288             nan     0.1000 190677.8600
    ##      6  2131833.8920             nan     0.1000 102844.1196
    ##      7  1964449.3433             nan     0.1000 183803.2337
    ##      8  1820010.0050             nan     0.1000 115022.8415
    ##      9  1643693.5164             nan     0.1000 85380.3652
    ##     10  1544818.7572             nan     0.1000 63230.4494
    ##     20   820337.3843             nan     0.1000 15842.3298
    ##     40   568138.9517             nan     0.1000 -5481.2074
    ##     60   478340.4862             nan     0.1000 -7744.6870
    ##     80   438202.0208             nan     0.1000 -14837.9028
    ##    100   403158.2862             nan     0.1000  984.4104
    ##    120   368678.0295             nan     0.1000 -5047.6475
    ##    140   342310.3113             nan     0.1000 -4666.3551
    ##    150   330989.1865             nan     0.1000 -3855.4625
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3568868.6929             nan     0.1000 308602.0521
    ##      2  3276474.3111             nan     0.1000 329069.5452
    ##      3  3020764.6074             nan     0.1000 266432.9549
    ##      4  2789809.9573             nan     0.1000 211417.7537
    ##      5  2602811.6084             nan     0.1000 144492.5935
    ##      6  2387714.5981             nan     0.1000 185168.2625
    ##      7  2204695.2977             nan     0.1000 148718.2898
    ##      8  2042499.8794             nan     0.1000 179801.1304
    ##      9  1912454.4944             nan     0.1000 153945.0400
    ##     10  1822821.1790             nan     0.1000 95111.0836
    ##     20  1007122.6438             nan     0.1000 51393.7659
    ##     40   644718.9837             nan     0.1000  287.1124
    ##     60   560920.7309             nan     0.1000  151.5161
    ##     80   523549.0177             nan     0.1000 -12091.2376
    ##    100   501498.6755             nan     0.1000 -1450.0832
    ##    120   452164.7491             nan     0.1000 -4735.8090
    ##    140   430085.7040             nan     0.1000 -21235.7323
    ##    150   420612.4663             nan     0.1000 -3282.2043
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3581774.6042             nan     0.1000 334936.3144
    ##      2  3217575.4356             nan     0.1000 378841.6451
    ##      3  2926205.3589             nan     0.1000 250176.2661
    ##      4  2673671.4472             nan     0.1000 234122.2956
    ##      5  2422892.6761             nan     0.1000 220428.0022
    ##      6  2254817.1245             nan     0.1000 185399.4674
    ##      7  2050555.8448             nan     0.1000 161122.3877
    ##      8  1867904.5021             nan     0.1000 85240.1861
    ##      9  1729807.5980             nan     0.1000 146774.7640
    ##     10  1621408.5870             nan     0.1000 93254.8909
    ##     20   907243.0598             nan     0.1000 27049.7091
    ##     40   581630.0465             nan     0.1000 -4907.7689
    ##     60   467397.9494             nan     0.1000 -484.2991
    ##     80   430393.6993             nan     0.1000 -20585.3614
    ##    100   399476.7027             nan     0.1000 -11642.8724
    ##    120   367264.4965             nan     0.1000 -3216.2857
    ##    140   344518.7832             nan     0.1000 -10937.4475
    ##    150   334819.9228             nan     0.1000 -3366.9818
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3426831.3252             nan     0.1000 552827.3107
    ##      2  3060415.8091             nan     0.1000 253596.8891
    ##      3  2773968.4741             nan     0.1000 300158.0082
    ##      4  2563031.6352             nan     0.1000 181218.2575
    ##      5  2358590.2385             nan     0.1000 95241.6476
    ##      6  2085658.3398             nan     0.1000 286850.2499
    ##      7  1897818.4253             nan     0.1000 150371.3590
    ##      8  1754351.2461             nan     0.1000 107511.8277
    ##      9  1578309.7834             nan     0.1000 95430.0992
    ##     10  1418280.9204             nan     0.1000 149238.9000
    ##     20   845595.7647             nan     0.1000 2720.6474
    ##     40   558091.1826             nan     0.1000 -7781.8371
    ##     60   501496.0994             nan     0.1000 -12328.0980
    ##     80   470988.5069             nan     0.1000 -4239.5698
    ##    100   445987.8851             nan     0.1000 -11328.0354
    ##    120   420637.9096             nan     0.1000 -11284.0342
    ##    140   379936.4627             nan     0.1000 -18506.0290
    ##    150   368987.9330             nan     0.1000 -4443.3449
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3400341.1689             nan     0.1000 263432.9524
    ##      2  3132766.7391             nan     0.1000 198622.8859
    ##      3  2878979.3006             nan     0.1000 164131.2002
    ##      4  2558858.9439             nan     0.1000 285558.3968
    ##      5  2345980.6715             nan     0.1000 225437.4607
    ##      6  2195354.1367             nan     0.1000 104394.7869
    ##      7  2097090.8460             nan     0.1000 61901.2908
    ##      8  1941428.4058             nan     0.1000 180207.2417
    ##      9  1768835.8006             nan     0.1000 143861.6938
    ##     10  1674766.8650             nan     0.1000 24323.4425
    ##     20   981920.9139             nan     0.1000 -1309.9925
    ##     40   615713.9252             nan     0.1000 -2328.5088
    ##     60   531785.2704             nan     0.1000 -9733.5852
    ##     80   493987.7029             nan     0.1000 -8893.8892
    ##    100   458270.2467             nan     0.1000 -4509.2140
    ##    120   438257.1633             nan     0.1000 -4641.9900
    ##    140   419766.1237             nan     0.1000 -8051.1558
    ##    150   412898.6775             nan     0.1000 -4831.0305
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3490784.6965             nan     0.1000 292878.6922
    ##      2  3158740.6644             nan     0.1000 289821.4236
    ##      3  2825849.4263             nan     0.1000 210646.1184
    ##      4  2450805.9547             nan     0.1000 290423.2390
    ##      5  2258373.4573             nan     0.1000 198854.9357
    ##      6  2035159.4378             nan     0.1000 122995.1935
    ##      7  1869515.9869             nan     0.1000 125411.4089
    ##      8  1695008.9031             nan     0.1000 141720.4366
    ##      9  1603593.7026             nan     0.1000 94167.4484
    ##     10  1482499.9102             nan     0.1000 123756.0798
    ##     20   845923.9503             nan     0.1000 14427.4156
    ##     40   589582.3043             nan     0.1000 -283.5009
    ##     60   511087.3683             nan     0.1000 -3791.0031
    ##     80   469836.7112             nan     0.1000 -4906.2823
    ##    100   427611.6916             nan     0.1000 -8651.2128
    ##    120   398312.5453             nan     0.1000 -4263.3524
    ##    140   377596.5196             nan     0.1000 -7695.2800
    ##    150   365378.1862             nan     0.1000 -7217.6240
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3459853.7572             nan     0.1000 298246.8378
    ##      2  3093116.6330             nan     0.1000 271972.5705
    ##      3  2780156.1541             nan     0.1000 278057.6479
    ##      4  2491495.8323             nan     0.1000 153853.5950
    ##      5  2266660.9063             nan     0.1000 200093.3519
    ##      6  2104339.4054             nan     0.1000 170062.5475
    ##      7  1963676.8381             nan     0.1000 142309.2700
    ##      8  1832003.7694             nan     0.1000 112636.0772
    ##      9  1702474.9184             nan     0.1000 124662.6769
    ##     10  1567863.6332             nan     0.1000 127827.5119
    ##     20   881108.0898             nan     0.1000 28508.6673
    ##     40   612852.6787             nan     0.1000 1189.5854
    ##     60   513646.7806             nan     0.1000 -8346.6726
    ##     80   453004.3439             nan     0.1000 -14202.7489
    ##    100   397109.9207             nan     0.1000 -9765.5786
    ##    120   361295.2081             nan     0.1000 -13705.1455
    ##    140   341673.7985             nan     0.1000 -6687.2198
    ##    150   330667.5999             nan     0.1000 -12235.4752
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1  3440430.2489             nan     0.1000 314749.7592
    ##      2  3196405.1541             nan     0.1000 287358.7494
    ##      3  2894002.3663             nan     0.1000 218627.9928
    ##      4  2684181.4533             nan     0.1000 171103.3005
    ##      5  2417238.9703             nan     0.1000 230665.6033
    ##      6  2239927.9059             nan     0.1000 187392.8842
    ##      7  2132397.5627             nan     0.1000 80151.3635
    ##      8  1958835.9003             nan     0.1000 154072.5975
    ##      9  1823066.4836             nan     0.1000 105990.5548
    ##     10  1668336.1622             nan     0.1000 86495.8601
    ##     20   970930.6068             nan     0.1000 43523.1924
    ##     40   599569.0427             nan     0.1000 6384.0197
    ##     60   506273.5657             nan     0.1000 -9989.3076
    ##     80   449022.3364             nan     0.1000 -2895.6046
    ##    100   420406.8695             nan     0.1000 -6993.7777
    ##    120   398929.7935             nan     0.1000 -4163.0544
    ##    140   369189.9745             nan     0.1000   81.2088
    ##    150   358233.2040             nan     0.1000 -2785.2728

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
    ##   n.trees     RMSE  Rsquared      MAE   RMSESD
    ## 1      50 844.1305 0.8736937 708.4486 257.7602
    ## 4      50 886.8888 0.8522844 727.8371 240.1523
    ## 7      50 882.1262 0.8562013 725.0814 263.7859
    ## 2     100 832.0928 0.8726336 696.2348 240.6215
    ## 5     100 890.4470 0.8486615 719.0385 201.4186
    ## 8     100 879.1062 0.8504655 715.0794 245.1105
    ## 3     150 830.2909 0.8763369 690.4309 236.7493
    ## 6     150 916.3451 0.8421880 745.4908 173.1569
    ## 9     150 887.3243 0.8472369 729.8468 216.9163
    ##   RsquaredSD    MAESD
    ## 1 0.06056947 242.1558
    ## 4 0.06220548 215.7860
    ## 7 0.05722550 242.0016
    ## 2 0.05454529 215.3977
    ## 5 0.05837817 181.9370
    ## 8 0.05811878 216.6798
    ## 3 0.05368148 208.6918
    ## 6 0.05702456 155.0466
    ## 9 0.04621794 193.5819

Now, we make predictions on the test data sets using the best model
fits. Then we compare RMSE to determine the best model.

``` r
predTree <- predict(fitTree, newdata = select(dayTest, -cnt))
postResample(predTree, dayTest$cnt)
```

    ##         RMSE     Rsquared          MAE 
    ## 1272.5431392    0.5972237 1021.9454450

``` r
boostPred <- predict(fitBoost, newdata = select(dayTest, -cnt))
postResample(boostPred, dayTest$cnt)
```

    ##       RMSE   Rsquared        MAE 
    ## 944.280587   0.750086 739.047512

When we compare the two models, the boosted tree model have lower RMSE
values when applied on the test dataset.
