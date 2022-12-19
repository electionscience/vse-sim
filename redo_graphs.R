library(data.table)
library(scatterD3)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(forcats)
library(scales)
library(extrafont)
#remotes::install_version("Rttf2pt1", version = "1.3.8")
#extrafont::font_import()
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
star_palette = c("#ee2c53", "#ff9900", "#60b33c", "#2aa2b3", "#02627c", "#4d2586", "#000000", "#c4e44c")
correct_order = rev(c("STAR", "Minimax", "ApprovalTop2", "Approval", "IRV", "PluralityTop2", "Plurality"))
correct_labels = rev(c("STAR", "Smith/Minimax", "Approval Top Two", "Approval", "IRV", "Plurality Top Two", "Plurality"))
model_names = c("base_scenario", "impartial_culture", "2d_model", "3d_model", "3cand", "8cand", "plurality_picky", "no_noise",
                "nostrats")

#modelName = "base_scenario"
for (modelName in model_names) {
  svgfiles = Sys.glob(paste0(modelName,"*.svg"))
  if (length(svgfiles) == 3) {
    next
  }
  files = Sys.glob(paste0(modelName,"*.csv"))
  print(files[1])
  fvse = fread(files[1])
  for (file in files[-1]) {
    print(file)
    fvse = rbind(fvse, fread(file))
  } 
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
                     fgArgs == "{'info': 'e'}",
       list(
         vse=mean((r1WinnerUtil - meanCandidateUtil) / (magicBestUtil - meanCandidateUtil)),
         fgMatters=mean(fgUtilDiff != 0),
         VSEDiff=mean((totalUtil - r1WinnerUtil) / (magicBestUtil - meanCandidateUtil)),
         #margStrategicRegret=mean(margStrategicRegret),
         pivotalUtilDiff=mean(pivotalUtilDiff)
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
         pivotalUtilDiff=mean(pivotalUtilDiff)
         ),
       by=.(method,backgroundStrat, fgStrat, fgArgs, fgTargets, intensityMaybe, fgBaseMaybe)]
  fromawares
  
  besttargets = function(grouped) {
    grouped[,target:=fgTargets[order(-pivotalUtilDiff)[1]], by=.(method,backgroundStrat, fgStrat, fgArgs)]
  #  justBest = grouped[target,,]
    chosen = grouped[grouped[, .I[pivotalUtilDiff == max(pivotalUtilDiff)], by = .(method,backgroundStrat, fgStrat, fgArgs)]$V1]
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
  vses %>% 
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
  + scale_color_manual(values=star_palette[c(7,2)], labels = c("Honest / Naive", "Viability-aware"))
    + theme_gdocs() 
      + theme(axis.title.y=element_blank())
      + xlab("% Voter Satisfaction Efficiency (VSE)")
      + labs(color="Voter Behavior") #+ scale_colour_colorblind(labels = c("Honest / Naive", "Viability-aware"))
     # + scale_y_discrete(breaks=c("STAR", "Plurality", "PluralityTop2", "Minimax", "Irv", "ApprovalTop2", "Approval"),
    #                     labels=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval"))
  ) 
  ggsave(paste0(modelName, " VSE.svg"),
         plot=vseGraph,
         width = 6.4, height = 2.4, dpi=1200, units = "in")
  # ggsave(paste0(modelName, " VSE.png"),
  #        plot=vseGraph,
  #        width = 6.4, height = 2.4, dpi=1200, units = "in")
  
  #correct_order = vses[,mean(VSE),by=method][order(V1),method]
  fromhons = fromhons[,method:=factor(method, levels=correct_order, labels = correct_labels)]
  #PVSI for viability-aware
  PVSI1 = (
    fromhons %>% 
      # mutate(method = method 
      #        #%>% fct_reorder(VSE, .fun='mean') 
      #        %>% recode(`STAR`="STAR", 
      #                   `PluralityTop2`="Plurality Top Two", 
      #                   `Minimax`="Smith/Minimax", 
      #                   `Irv`="IRV (RCV)", 
      #                   `ApprovalTop2`="Approval Top Two", 
      #        ),
      #        #to=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval")))
      #        #VSE = VSE * 100
      # ) %>%
      ggplot(
        aes(x = pivotalUtilDiff, y = as.factor(method))) 
    + scale_x_continuous(labels=scales::percent_format(accuracy = 1))#, limits = c(-.0,.38))
    + geom_point(size=3, color=star_palette[2]) #+ xlim(.65,1.00) 
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
  # ggsave(paste0(modelName, " PVSI1.png"),
  #        plot=PVSI1,
  #        width = 7.8, height = 2.6, dpi=1200, units = "in")
  
  bestfromawares = bestfromawares[fgBaseMaybe == "p" | fgStrat=="bulletBallot",] 
  
  bestfromawares[intensityMaybe == "3" & fgStrat=="compBallot",xStrategy:="Favorite Betrayal"]
  bestfromawares[intensityMaybe == "3" & fgStrat=="diehardBallot",xStrategy:="Burial"]
  bestfromawares[fgStrat=="bulletBallot",xStrategy:="Bullet Voting*"]
  bestfromawares[(intensityMaybe %in% c("1","2")) & fgStrat=="diehardBallot" &
                   (method %in% c("Approval", "ApprovalTop2")),xStrategy:="Exclusive"]
  bestfromawares[(intensityMaybe %in% c("1","2")) & fgStrat=="compBallot" &
                   (method %in% c("Approval", "ApprovalTop2")),xStrategy:="Inclusive"]
  bestfromawares[(intensityMaybe == "2") & fgStrat=="compBallot" &
                   (method %in% c("STAR","Minimax")),xStrategy:="Polarized Inclusive"]
  bestfromawares[(intensityMaybe == "2") & fgStrat=="diehardBallot" &
                   (method %in% c("STAR","Minimax")),xStrategy:="Polarized Exclusive"]
  bestfromawares[(intensityMaybe == "1") & fgStrat=="diehardBallot" &
                   (method %in% c("STAR","Minimax")),xStrategy:="Honest Deflation"]
  bestfromawares[(intensityMaybe == "1") & fgStrat=="compBallot" &
                   (method %in% c("STAR","Minimax")),xStrategy:="Honest Inflation"]
  bestfromawares[startsWith(fgArgs,"{'fallback': 'va'"),xStrategy:=NA]
  fixedfromawares = bestfromawares[!is.na(xStrategy) & 
                                     !(fgStrat=="bulletBallot" & method %in% c("Plurality", "PluralityTop2")) &
                                     !(xStrategy=="Favorite Betrayal" & method == "Approval")
                                     ,] 
  
  strategies = sort(unique(fixedfromawares[,xStrategy]))
  strategies
  fixedfromawares[,Strategy:=factor(xStrategy, levels = strategies[c(4,2,1,3,7,8,9,5,6)])]
  fixedfromawares[,method:=factor(method, levels=correct_order, labels = correct_labels)]
  #methods = sort(unique(bestfromawares[,method]))
  #methodsInOrder = methods[c(5,6,3,1,2,4,7)]
  #methodsInOrder # should be: [1] "Plurality"     "PluralityTop2" "IRV"           "Approval"      "ApprovalTop2"  "Minimax"       "STAR"  
  #bestfromawares[,method:=factor(method, levels = methodsInOrder)]
  
  #PVSI for targeted strategy
  PVSI2 = (
    fixedfromawares %>% 
      # mutate(method = method 
      #        #%>% fct_reorder(VSE, .fun='mean') 
      #        %>% recode(`STAR`="STAR", 
      #                   `PluralityTop2`="Plurality Top Two", 
      #                   `Minimax`="Smith/Minimax", 
      #                   `Irv`="IRV (RCV)", 
      #                   `ApprovalTop2`="Approval Top Two", 
      #        ),
      #        #to=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval")))
      #        #VSE = VSE * 100
      # ) %>%
      ggplot(
        aes(x = pivotalUtilDiff, y = as.factor(method), color = Strategy, fill=Strategy, shape=Strategy)) 
    + scale_x_continuous(labels=scales::percent_format(accuracy = 1))
    + geom_point(size=3) #+ xlim(.65,1.00) 
    #+ scale_shape_manual(values=c(0,1,5,2,6))
    + scale_shape_manual(values=c(21,22,23,25,24,25,24,25,24))
    + scale_color_manual(values=star_palette[c(1,2,3,4,4,5,5,6,6)])
    + scale_fill_manual(values=star_palette[c(1,2,8,4,4,5,5,6,6)])
    + theme_gdocs() 
    + theme(axis.title.y=element_blank()) + xlab("% Pivotal Voter Strategic Incentive (PVSI)")
    + labs(color="Strategy") 
    # + scale_y_discrete(breaks=c("STAR", "Plurality", "PluralityTop2", "Minimax", "Irv", "ApprovalTop2", "Approval"),
    #                     labels=c("STAR", "Plurality/Runoff", "Plurality", "Smith/Minimax", irv="IRV (RCV)", "Approval/Runoff", "Approval"))
  ) 
  ggsave(paste0(modelName, " PVSI2.svg"),
         plot=PVSI2,
         width = 9.6, height = 3.1, dpi=1200, units = "in")
  
  
  print("Did it!")
  print(modelName)
  print(modelName)
  print(modelName)
  print(modelName)
  print(modelName)
  print(modelName)
  print(modelName)
}

