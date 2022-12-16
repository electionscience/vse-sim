library(data.table)
library(scatterD3)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(forcats)
library(scales)
library(extrafont)
#remotes::install_version("Rttf2pt1", version = "1.3.8")
extrafont::font_import()
library(showtext)
font_add("Arial", "/Library/Fonts/Arial.ttf")  
# 
# vse = fread("full1.csv")
# vse = rbind(vse,fread("full2.csv"))
# vse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]
# 
# vse[chooser=="honBallot",mean(util-rand)/mean(best-rand),by=list(method,chooser)]
# 
# ovse = fread("lowDks2.csv")
# ovse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]
# 
# ovse[chooser %in% c("honBallot","Oss.hon_strat.","Prob.strat50_hon50."),mean(util-rand)/mean(best-rand),by=list(method,chooser)]
# 
# mvse = fread("media413.csv")
# mvse = rbind(mvse,fread("media41.csv"), fill=T)
# mvse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]
# 
# mvse[chooser %in% c("honBallot","Oss.hon_strat.","Prob.strat50_hon50."),mean(util-rand)/mean(best-rand),by=list(method,chooser)]
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# fvse = fread("fuzzy5.csv")
# fvse[,mean(util-rand)/mean(best-rand),by=list(method,chooser)]
# 
# fvse = fread("wtf1.csv")
# 
# fvse = rbind(fvse,fread("wtf2.csv"))

modelName = "picky.7.pe.0.1.1"
fvse = fread(paste0(modelName, ".csv"))
numVoters = mean(fvse[,numVoters])
vses = fvse[method != "ApprovalPoll",list(VSE=mean((r1WinnerUtil - meanCandidateUtil) / 
                      (magicBestUtil - meanCandidateUtil))),by=.(method,backgroundStrat)]
dcast(vses, method ~ backgroundStrat)
vses

fromhons = fvse[backgroundStrat=="honBallot" & fgStrat == "vaBallot" & 
                  method != "Minimax" & fgArgs == "{'info': 'e'}",
     list(
       vse=mean((r1WinnerUtil - meanCandidateUtil) / (magicBestUtil - meanCandidateUtil)),
       fgMatters=mean(fgUtilDiff != 0),
       VSEDiff=mean((totalUtil - r1WinnerUtil) / (magicBestUtil - meanCandidateUtil)),
       #margStrategicRegret=mean(margStrategicRegret),
       avgStrategicRegret=mean((avgStrategicRegret) / (magicBestUtil - meanCandidateUtil) / numVoters),
       fgHelpedUtilDiff=mean((fgHelpedUtilDiff) / (magicBestUtil - meanCandidateUtil) / numVoters),
       fgHarmedUtilDiff=mean((fgHarmedUtilDiff) / (magicBestUtil - meanCandidateUtil) / numVoters)
     ),
     by=.(method,backgroundStrat, fgStrat, fgArgs)]
fromhons

fromawares = fvse[((backgroundStrat=="vaBallot"  & method != "Minimax") | (backgroundStrat=="honBallot" &method == "Minimax")) 
                  & !fgStrat %in% c("", "vaBallot") & !startsWith(fgArgs, "{'fallback': 'hon',"),
     list(
       vse=mean((r1WinnerUtil - meanCandidateUtil) / (magicBestUtil - meanCandidateUtil)),
       fgMatters=mean(fgUtilDiff != 0),
       VSEDiff=mean((totalUtil - r1WinnerUtil) / (magicBestUtil - meanCandidateUtil)),
       #margStrategicRegret=mean(margStrategicRegret),
       avgStrategicRegret=mean(avgStrategicRegret / (magicBestUtil - meanCandidateUtil) / numVoters),
       fgHelpedUtilDiff=mean(fgHelpedUtilDiff  / (magicBestUtil - meanCandidateUtil) / numVoters),
       fgHarmedUtilDiff=mean(fgHarmedUtilDiff / (magicBestUtil - meanCandidateUtil) / numVoters)),
     by=.(method,backgroundStrat, fgStrat, fgArgs, fgTargets)]
fromawares

besttargets = function(grouped) {
  grouped[,target:=fgTargets[order(-avgStrategicRegret)[1]], by=.(method,backgroundStrat, fgStrat, fgArgs)]
#  justBest = grouped[target,,]
  chosen = grouped[grouped[, .I[avgStrategicRegret == max(avgStrategicRegret)], by = .(method,backgroundStrat, fgStrat, fgArgs)]$V1]
}
bestfromawares = besttargets(fromawares)
bestfromawares
fixedFavoriteBetrayal = fromawares[method=="STAR" & fgTargets == "select31" & fgStrat == "compBallot" & fgArgs == "{'intensity': 3}",]
removeWrongFB = bestfromawares[!(fgStrat == "compBallot" & fgArgs == "{'intensity': 3}" & method=="STAR"),]
bestfromawares = rbind(fixedFavoriteBetrayal, removeWrongFB)

#vse as bars
(ggplot(data = vses, aes(x = VSE, y = method, group = method)) 
  + geom_line(size=3) #+ xlim(.65,1.00) 
  + theme_gdocs() 
  + theme(axis.title.y=element_blank()) + xlab("% Voter Satisfaction Efficiency (VSE)")) 

#vse as dots
vseGraph = (
vses[!(method=="Minimax" & backgroundStrat=="vaBallot"),] %>% 
    mutate(method = method 
                    %>% fct_reorder(VSE, .fun='mean') 
                    %>% recode(`STAR`="STAR", 
                               `PluralityTop2`="Plurality Top Two", 
                               `Minimax`="Smith/Minimax", 
                               `Irv`="IRV (RCV)", 
                               `ApprovalTop2`="Approval Top Two", 
                               ),
                                  #to=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval")))
           #VSE = VSE * 100
           ) %>%
    ggplot(
          aes(x = VSE, y = as.factor(method), color = backgroundStrat)) 
  + scale_x_continuous(labels=scales::percent_format(accuracy = 1))#, limits=c(.68, 1.0))
    + geom_point(size=3) #+ xlim(.65,1.00) 
  + theme_gdocs() 
    + theme(axis.title.y=element_blank())
    + xlab("% Voter Satisfaction Efficiency (VSE)")
    + labs(color="Voter Behavior") + scale_colour_colorblind(labels = c("Honest / Naive", "Viability-aware"))
   # + scale_y_discrete(breaks=c("STAR", "Plurality", "PluralityTop2", "Minimax", "Irv", "ApprovalTop2", "Approval"),
  #                     labels=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval"))
) 
ggsave(paste0(modelName, " VSE.svg"),
       plot=vseGraph,
       width = 6.4, height = 2.4, dpi=1200, units = "in")
ggsave(paste0(modelName, " VSE.png"),
       plot=vseGraph,
       width = 6.4, height = 2.4, dpi=1200, units = "in")

#ASI for viability-aware
ASI1 = (
  fromhons %>% 
    mutate(method = method 
           %>% fct_reorder(avgStrategicRegret, .fun='mean') 
           %>% recode(`STAR`="STAR", 
                      `PluralityTop2`="Plurality Top Two", 
                      `Minimax`="Smith/Minimax", 
                      `Irv`="IRV (RCV)", 
                      `ApprovalTop2`="Approval Top Two", 
           ),
           #to=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval")))
           #VSE = VSE * 100
    ) %>%
    ggplot(
      aes(x = avgStrategicRegret, y = as.factor(method))) 
  + scale_x_continuous(labels=scales::percent_format(accuracy = 1))#, limits = c(-.0,.38))
  + geom_point(size=3) #+ xlim(.65,1.00) 
  + theme_gdocs() 
  + theme(axis.title.y=element_blank()) + xlab("% Average Strategic Incentive (ASI)")
  #+ labs(color="Voter Behavior") 
  #+ scale_colour_colorblind(labels = c("Honest / Naive", "Viability-aware"))
  # + scale_y_discrete(breaks=c("STAR", "Plurality", "PluralityTop2", "Minimax", "Irv", "ApprovalTop2", "Approval"),
  #                     labels=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval"))
) 

ggsave(paste0(modelName, " ASI1.svg"),
       plot=ASI1,
       width = 7.8, height = 2.6, dpi=1200, units = "in")
ggsave(paste0(modelName, " ASI1.png"),
       plot=ASI1,
       width = 7.8, height = 2.6, dpi=1200, units = "in")

bestfromawares[fgArgs == "{'intensity': 3}" & fgStrat=="compBallot",xStrategy:="Favorite Betrayal"]
bestfromawares[fgArgs == "{'intensity': 3}" & fgStrat=="diehardBallot",xStrategy:="Burial"]
bestfromawares[fgArgs == "{'intensity': 4}",xStrategy:="Bullet (random bloc)"]
bestfromawares[(fgArgs %in% c("{'intensity': 1}","{'intensity': 2}")) & fgStrat=="diehardBallot",xStrategy:="Exclusive Approval-like"]
bestfromawares[(fgArgs %in% c("{'intensity': 1}","{'intensity': 2}")) & fgStrat=="compBallot",xStrategy:="Inclusive Approval-like"]
bestfromawares = bestfromawares[!(method=="STAR" & fgArgs == "{'intensity': 1}"),] 
bestfromawares[startsWith(fgArgs,"{'fallback': 'va'"),xStrategy:="Bullet (one of top 3)"]

strategies = sort(unique(bestfromawares[,xStrategy]))
bestfromawares[,Strategy:=factor(xStrategy, levels = strategies[c(3,5,1,4,6,2)])]

#ASI for targeted strategy
ASI2 = (
  bestfromawares %>% 
    mutate(method = method 
           %>% fct_reorder(avgStrategicRegret, .fun='mean') 
           %>% recode(`STAR`="STAR", 
                      `PluralityTop2`="Plurality Top Two", 
                      `Minimax`="Smith/Minimax", 
                      `Irv`="IRV (RCV)", 
                      `ApprovalTop2`="Approval Top Two", 
           ),
           #to=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval")))
           #VSE = VSE * 100
    ) %>%
    ggplot(
      aes(x = avgStrategicRegret, y = as.factor(method), color = Strategy, fill=Strategy, shape=Strategy)) 
  + scale_x_continuous(labels=scales::percent_format(accuracy = 1))
  + geom_point(size=3) #+ xlim(.65,1.00) 
  #+ scale_shape_manual(values=c(0,1,5,2,6))
  + scale_shape_manual(values=c(21:25,1))
  + theme_gdocs() 
  + theme(axis.title.y=element_blank()) + xlab("% Average Strategic Incentive (ASI)")
  + labs(color="Strategy") 
  # + scale_y_discrete(breaks=c("STAR", "Plurality", "PluralityTop2", "Minimax", "Irv", "ApprovalTop2", "Approval"),
  #                     labels=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval"))
) 
ggsave(paste0(modelName, " ASI2.svg"),
       plot=ASI2,
       width = 8.4, height = 2.8, dpi=1200, units = "in")

