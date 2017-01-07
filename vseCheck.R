library(data.table)

vse = fread("ksresults3.csv")
vse[,mean(vse),by=list(method,chooser)]