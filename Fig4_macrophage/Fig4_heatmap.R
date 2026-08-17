DEG_path = 'your path'

name = 'cd68-in'
DEG1 = read.csv(paste0(DEG_path,'/', name, '/control-OW.csv'))
DEG1 = DEG1[order(DEG1[,5]), ]
DEG2 = read.csv(paste0(DEG_path,'/', name, '/OW-T2D.csv'))
DEG2 = DEG2[order(DEG2[,5]), ]
DEG3 = read.csv(paste0(DEG_path,'/', name, '/control-T2D.csv'))
DEG3 = DEG3[order(DEG3[,5]), ]
colnames(DEG1)[1] = 'id'
colnames(DEG2)[1] = 'id'
colnames(DEG3)[1] = 'id'
DEG = rbind(DEG1[,1:2], DEG2[,1:2], DEG3[,1:2])
fc = abs(DEG[,2])
use_DEG = DEG[order(fc, decreasing = TRUE),]
uni_DEG = unique(use_DEG[,1])
use_DEG = use_DEG[match(uni_DEG,use_DEG[,1]),]

a = match(use_DEG[,1], pri_list)
b = is.na(a)
use_DEG1 = use_DEG[b,]



name = 'cd68-out'
DEG1 = read.csv(paste0(DEG_path,'/', name, '/control-OW.csv'))
DEG1 = DEG1[order(DEG1[,5]), ]
DEG2 = read.csv(paste0(DEG_path,'/', name, '/OW-T2D.csv'))
DEG2 = DEG2[order(DEG2[,5]), ]
DEG3 = read.csv(paste0(DEG_path,'/', name, '/control-T2D.csv'))
DEG3 = DEG3[order(DEG3[,5]), ]
colnames(DEG1)[1] = 'id'
colnames(DEG2)[1] = 'id'
colnames(DEG3)[1] = 'id'
DEG = rbind(DEG1[,1:2], DEG2[,1:2], DEG3[,1:2])
fc = abs(DEG[,2])
use_DEG = DEG[order(fc, decreasing = TRUE),]
uni_DEG = unique(use_DEG[,1])
use_DEG = use_DEG[match(uni_DEG,use_DEG[,1]),]


b = is.na(a)
use_DEG2 = use_DEG[b,]


use_DEG = rbind(use_DEG1, use_DEG2)
fc = abs(use_DEG[,2])
use_DEG = use_DEG[order(fc, decreasing = TRUE),]
uni_DEG = unique(use_DEG[,1])
use_DEG = use_DEG[match(uni_DEG,use_DEG[,1]),]

DEG = use_DEG[1:100,1]

a = which(segment_all == 'cd68-in')
b = which(a>0)

celltype1 = segment_all[a[b]]
group1 = group_all[a[b]]
data1 = exp_all[DEG,a[b]]

a = which(segment_all == 'cd68-out')
b = which(a>0)

celltype2 = segment_all[a[b]]
group2 = group_all[a[b]]
data2 = exp_all[DEG,a[b]]

data = cbind(data1, data2)
group = c(group1,group2)
celltype = c(celltype1, celltype2)

library(pheatmap)
library(ggplot2)
library(RColorBrewer)

colors <- colorRampPalette(c("deepskyblue", "white", "darkred"))(length(color_breaks) - 1)


data <- data[, order(group)]
celltype <- celltype[order(group)] 
group <- group[order(group)]

group_colors = c("control" = "#00CC00", "OW" = "#619CFF", "T2D" = "#F8766D")
celltype_colors = c("cd68-in" = "#00008B", "cd68-out" = "#FFA500")


annotation_col = data.frame(
  CellType = factor(celltype, levels = c('cd68-in', 'cd68-out')),  
  State = factor(group, levels = c('control', 'OW', 'T2D'))  
  
)
rownames(annotation_col) = colnames(data)  


annotation_colors = list(
  CellType = celltype_colors,  
  State = group_colors  
  
)


p1 <- pheatmap(data,
               color = colorRampPalette(c("deepskyblue", "white", "darkred"))(100),  
               border_color = "black", 
               scale = 'row',  
               cluster_rows = TRUE,  
               cluster_cols = FALSE, 
               treeheight_row = 0,  
               treeheight_col = 0,  
               legend = TRUE,  
               legend_breaks = c(-4, 0, 4),  
               legend_labels = c("low", "", "high"),  
               show_rownames = TRUE, 
               show_colnames = FALSE, 
               fontsize = 15,  
               fontsize_row = 7,  
               fontsize_col = 14,  
               number_color = "black", 
               fontsize_number = 30,  
               cellheight = 7, 
               cellwidth = 1,  
               annotation_legend = TRUE,
               annotation_names_row = TRUE,  
               annotation_col = annotation_col,  
               annotation_colors = annotation_colors  
)

p1




