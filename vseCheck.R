library(data.table)

vse = fread("full1.csv")
vse = rbind(vse,fread("full2.csv"))
vse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]

vse[chooser=="honBallot",mean(util-rand)/mean(best-rand),by=list(method,chooser)]

ovse = fread("lowDks2.csv")
ovse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]

ovse[chooser %in% c("honBallot","Oss.hon_strat.","Prob.strat50_hon50."),mean(util-rand)/mean(best-rand),by=list(method,chooser)]

mvse = fread("media413.csv")
mvse = rbind(mvse,fread("media41.csv"), fill=T)
mvse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]

mvse[chooser %in% c("honBallot","Oss.hon_strat.","Prob.strat50_hon50."),mean(util-rand)/mean(best-rand),by=list(method,chooser)]


fvse = fread("fuzzy1.csv")
fvse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]
