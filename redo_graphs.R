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

modelName = "base_noise_0.1_"
files = Sys.glob(paste0(modelName,"*.csv"))
fileName = sort(files, decreasing = T)[1]
fvse = fread(fileName)
numVoters = mean(fvse[,numVoters])
vses = fvse[method != "ApprovalPoll",list(VSE=mean((r1WinnerUtil - meanCandidateUtil) / 
                      (magicBestUtil - meanCandidateUtil))),by=.(method,backgroundStrat)]
dcast(vses, method ~ backgroundStrat)
vses

fvse = fvse[,intensityMaybe:=substr(fgArgs,15,15)]
unique(fvse[,intensityMaybe]) #Should be: [1] ""  "1" "2" "3" "4" "h" "v"
fvse = fvse[,fgBaseMaybe:=substr(fgArgs,27,27)]
unique(fvse[,fgBaseMaybe]) #Should be: [1] ""  "e" "p" ":" " "

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
     by=.(method,backgroundStrat, fgStrat, fgArgs, intensityMaybe, fgBaseMaybe)]
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
     by=.(method,backgroundStrat, fgStrat, fgArgs, fgTargets, intensityMaybe, fgBaseMaybe)]
fromawares

besttargets = function(grouped) {
  grouped[,target:=fgTargets[order(-avgStrategicRegret)[1]], by=.(method,backgroundStrat, fgStrat, fgArgs)]
#  justBest = grouped[target,,]
  chosen = grouped[grouped[, .I[avgStrategicRegret == max(avgStrategicRegret)], by = .(method,backgroundStrat, fgStrat, fgArgs)]$V1]
}
bestfromawares = besttargets(fromawares)
bestfromawares
fixedFavoriteBetrayal = fromawares[method=="STAR" & fgTargets == "select31" & fgStrat == "compBallot" & intensityMaybe == "3",]
removeWrongFB = bestfromawares[!(fgStrat == "compBallot" & intensityMaybe == "3" & method=="STAR"),]
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

#PVSI for viability-aware
PVSI1 = (
  fromhons %>% 
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
      aes(x = avgStrategicRegret, y = as.factor(method))) 
  + scale_x_continuous(labels=scales::percent_format(accuracy = 1))#, limits = c(-.0,.38))
  + geom_point(size=3) #+ xlim(.65,1.00) 
  + theme_gdocs() 
  + theme(axis.title.y=element_blank()) + xlab("% Pivotal Voter Strategic Incentive (PVSI)")
  #+ labs(color="Voter Behavior") 
  #+ scale_colour_colorblind(labels = c("Honest / Naive", "Viability-aware"))
  # + scale_y_discrete(breaks=c("STAR", "Plurality", "PluralityTop2", "Minimax", "Irv", "ApprovalTop2", "Approval"),
  #                     labels=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval"))
) 

ggsave(paste0(modelName, " PVSI1.svg"),
       plot=PVSI1,
       width = 7.8, height = 2.6, dpi=1200, units = "in")
ggsave(paste0(modelName, " PVSI1.png"),
       plot=PVSI1,
       width = 7.8, height = 2.6, dpi=1200, units = "in")

bestfromawares[intensityMaybe == "3" & fgStrat=="compBallot",xStrategy:="Favorite Betrayal"]
bestfromawares[intensityMaybe == "3" & fgStrat=="diehardBallot",xStrategy:="Burial"]
bestfromawares[intensityMaybe == "4",xStrategy:="Bullet (random bloc)"]
bestfromawares[(intensityMaybe %in% c("1","2")) & fgStrat=="diehardBallot" &
                 (method %in% c("Approval", "ApprovalTop2")),xStrategy:="Threshold (Exclusive)"]
bestfromawares[(intensityMaybe %in% c("1","2")) & fgStrat=="compBallot" &
                 (method %in% c("Approval", "ApprovalTop2")),xStrategy:="Threshold (Inclusive)"]
bestfromawares[(intensityMaybe == "1") & fgStrat=="diehardBallot" &
                 (method == "STAR"),xStrategy:="Exaggerate (Exclusive?)"]
bestfromawares[(intensityMaybe == "1") & fgStrat=="compBallot" &
                 (method == "STAR"),xStrategy:="Huh (Inclusive?)"]
bestfromawares[(intensityMaybe == "2") & fgStrat=="diehardBallot" &
                 (method == "STAR"),xStrategy:="Huh (Exclusive?)"]
bestfromawares[(intensityMaybe == "2") & fgStrat=="compBallot" &
                 (method == "STAR"),xStrategy:="Exaggerate (Inclusive?)"]
bestfromawares[startsWith(fgArgs,"{'fallback': 'va'"),xStrategy:="XXX"]
bestfromawares = bestfromawares[(!xStrategy=="XXX") & fgBaseMaybe == "p",] 

strategies = sort(unique(bestfromawares[,xStrategy]))
bestfromawares[,Strategy:=factor(xStrategy, levels = strategies[c(5,2,1,8,9,6,7,3,4)])]
#methods = sort(unique(bestfromawares[,method]))
#methodsInOrder = methods[c(5,6,3,1,2,4,7)]
#methodsInOrder # should be: [1] "Plurality"     "PluralityTop2" "IRV"           "Approval"      "ApprovalTop2"  "Minimax"       "STAR"  
#bestfromawares[,method:=factor(method, levels = methodsInOrder)]

#PVSI for targeted strategy
PVSI2 = (
  bestfromawares %>% 
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
      aes(x = avgStrategicRegret, y = as.factor(method), color = Strategy, fill=Strategy, shape=Strategy)) 
  + scale_x_continuous(labels=scales::percent_format(accuracy = 1))
  + geom_point(size=3) #+ xlim(.65,1.00) 
  #+ scale_shape_manual(values=c(0,1,5,2,6))
  + scale_shape_manual(values=c(21,22,23,25,24,25,24,25,24))
  + scale_color_manual(values=c(1,2,3,4,4,5,5,6,6))
  + scale_fill_manual(values=c(1,2,3,4,4,5,5,6,6))
  + theme_gdocs() 
  + theme(axis.title.y=element_blank()) + xlab("% Pivotal Voter Strategic Incentive (PVSI)")
  + labs(color="Strategy") 
  # + scale_y_discrete(breaks=c("STAR", "Plurality", "PluralityTop2", "Minimax", "Irv", "ApprovalTop2", "Approval"),
  #                     labels=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval"))
) 
ggsave(paste0(modelName, " PVSI2.svg"),
       plot=PVSI2,
       width = 8.4, height = 2.8, dpi=1200, units = "in")

