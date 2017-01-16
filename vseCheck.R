library(data.table)
library(scatterD3)

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


fvse = fread("fuzzy5.csv")
fvse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]

fvse = fread("fuzzySchulze2.csv")
fvse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]
etype = fvse[method=="Schulze" & chooser=="extraEvents",tallyVal0,by=eid]
names(etype) = c("eid","scenario")
setkey(etype,eid)
setkey(fvse,eid)
fvse=fvse[etype]

interestingStrats = c("honBallot","stratBallot","Oss.hon_strat.","Oss.hon_Prob.strat50_hon50..","Prob.strat50_hon50.")
honestScenarios = fvse[chooser %in% interestingStrats,list(vse=mean(util-rand)/mean(best-rand),frequency=.N/dim(etype)[1]),by=list(scenario,chooser,method)]
write.csv(honestScenarios,"byScenario.csv")
hmethodlist = honestScenarios[,method]
methods = unique(hmethodlist)
methodOrder = methods[c(7,11,12,8,9,4,3,2,1,6,5,10)]
scenarios = unique(fvse[,scenario])
scenarioFreq = honestScenarios[,list(freq=mean(frequency)),by=scenario]
setkey(scenarioFreq,scenario)
scenarioLabelBase2 = c("2. Easy\n(Cond #1 = Plur #1)", 
                      "5. Chicken dilemma\n(Cond #3 = Plur3 #1)",
                      "6. Other\n",
                      "4. Center squeeze\n(Cond #1 = Plur3 #3)",
                      "3. Spoiler\n(Cond #1 = Plur3 #1)",
                      "1. Condorcet cycle\n"
) 
scenarioLabelBase = c("2.Easy", 
                      "5.Chicken dilem.",
                      "6.Other",
                      "4.Ctr. squeeze",
                      "3.Spoiler",
                      "1.Cond. cycle"
) 
scenarioLabel = paste0(scenarioLabelBase," (~",round(scenarioFreq[scenarios,freq]*100),"%)")
honestScenarios[,method:=factor(hmethodlist,levels=methodOrder,labels=paste(c(paste0(" ",as.character(1:9)),as.character(10:12)),methodOrder,sep=". "))]
honestScenarios[,strategy:=factor(chooser, levels=interestingStrats,labels=c("100% honest",
                                                                             "100% strategic",
                                                                             "100% 1-sided strategy","50% 1-sided strategy","50% strategic"))]
honestScenarios[,`Scenario type`:=factor(scenario,levels=scenarios,labels=scenarioLabel)]
scatterD3(data = honestScenarios, x = vse, y = method, col_var = strategy, symbol_var = `Scenario type`, left_margin = 90, xlim=c(0.4,1.0), size_var=frequency)

