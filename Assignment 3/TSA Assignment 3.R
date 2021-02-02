rm(list = ls())
library(rmarkdown)
setwd("C:/Users/lucas/OneDrive/Dokumente/2. Semester/Time Series/Data")
#read in data and rename GDP variable for ease of use 
gdp_seas_adj <- read.csv("GDP_DE_seas_adj_Fed_StLouis.csv", sep = ";")
colnames(gdp_seas_adj)[2] <- "GDP"

#generate time series data within R
gdp <- gdp_seas_adj[2]


#computing quarterly growth rate
#get dataframe ready
gdp[, c(2, 3, 4)] <- 0
colnames(gdp)[c(2, 3, 4)] <- c("log_gdp", "growth_rate", "date")
#calc log gdp 
gdp[, 2] <- log(gdp[, 1])
for(i in 2:dim(gdp)[1]){
  gdp[i, "growth_rate"] <- (gdp[i, 2] - gdp[i - 1, 2]) * 100
}
gdp[, 4] <- gdp_seas_adj[, 1]
#note: for beautiful ggplot need a date variable to define it as aes(x = date, y = data)

#fitting arma model 
#think we need to define the start as last month of 1990 since R thinks that data is end of quarter 
#but it is beginning of quarter
#there might be an error here when declaring the time series

#WARNING!!!!
#rdocumentation says that if freqeuncy is set to 4 then start and end 
#define year and the quarter, i.e. c(1991, 1) would be correct start 
#and c(2017, 4) correct end 
gdp <- ts(gdp, frequency = 4, start = c(1991, 1), end = c(2017, 3))
#what metheod should we use? maybe check Mehdi's stata code
arima(gdp[, "growth_rate"], order = c(1, 0, 1))

#find best fit with loop 
ARMA_res <- list() #empty list to store results 
#set counter 
count <- 1 
#loop over p 
for(p in 0:3){
  #loop over q 
    for(q in 0:3){
      ARMA_res[[count]] <- arima(x = gdp[, "growth_rate"], order = c(p, 0 , q))
      count = count + 1
    }
}
#getting some warnings here... 

#calculate AIC values 
ARMA_aic <- sapply(ARMA_res, function(x) x$aic)
which(ARMA_aic == min(ARMA_aic))
ARMA_opt <- ARMA_res[[which(ARMA_aic == min(ARMA_aic))]]
#5th model -> p = 1, q = 0
#how does this make sense? No moving average because of seasonal adjustment? 

#plot autocorrelation of residuals (Hint 4) 
res <- residuals(ARMA_opt)
autocorr <- acf(res, plot = FALSE)



