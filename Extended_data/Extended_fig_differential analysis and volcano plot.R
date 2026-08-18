library(edgeR)
library(tidyverse)
library(RColorBrewer)

p = list()
do_seg = 'ins' #segment
a = which(segment_all == do_seg)
sub_exp = exp_all[,a]
group = group_all[a,1]
condition = data.frame(con1 = c('control','obesity','control'),con2 = c('obesity','T2D','T2D'))


sub_exp <- sub_exp[!rownames(sub_exp) %in% filtered_gene,]

for (i in 1:3) {
  b = which(group == condition[i,1] | group == condition[i,2])
  group2 = group[b]
  group2 = as.character(group2)
  group_list=factor(group2,levels = unique(group2))
  table(group_list)
  express_cpm = sub_exp[,b]
  
  express_cpm[express_cpm<0] = 0
  exprSet <- as.matrix(express_cpm)
  design <- model.matrix(~0+factor(group_list))
  rownames(design) <- colnames(exprSet)
  colnames(design) <- levels(factor(group_list))
  
  DEG <- DGEList(counts=exprSet,  
                 group=factor(group_list))
  DEG$samples$lib.size
  DEG <- calcNormFactors(DEG)
  DEG$samples$norm.factors
  

  DEG <- estimateGLMCommonDisp(DEG,design)
  DEG <- estimateGLMTrendedDisp(DEG, design)
  DEG <- estimateGLMTagwiseDisp(DEG, design)
  
  fit <- glmFit(DEG, design)
  lrt <- glmLRT(fit, contrast=c(1,-1))

  DEG_edgeR <- as.data.frame(topTags(lrt, n=nrow(DEG)))
  head(DEG_edgeR)
  fc <- 0
  p <- 0.05
  DEG_edgeR$regulated <- ifelse(DEG_edgeR$logFC>log2(fc) & DEG_edgeR$PValue<p,
                                "down",ifelse(DEG_edgeR$logFC<(-log2(fc)) & DEG_edgeR$PValue<p,"up","not significant"))
  DEG_edgeR[,1] = -DEG_edgeR[,1]
  print(length(which(DEG_edgeR[,'regulated'] == 'up')))
  print(length(which(DEG_edgeR[,'regulated'] == 'down')))
  chayi = DEG_edgeR[which(DEG_edgeR$regulated != 'normal'),]
  write.csv(DEG_edgeR, paste0('your path',do_seg,'_',condition[i,1],'-',condition[i,2],'.csv'))
  
  
  paired_DEG_con <- read.csv("your path", row.names = 'X')
  paried_DEG_obe <- read.csv("your path", row.names = 'X')
  paired_DEG_t2d <- read.csv("your path", row.names = 'X')

  
  data <- paired_DEG_con
  data[,'gene_name'] = row.names(data)
  colnames(data)
  significant <- data$PValue < 0.05
  log2FC <- ifelse(data$logFC > 0 | data$logFC < 0, TRUE, FALSE)
  
  library(ggrepel)
  library(ggplot2)
  
  xlimit <- c(-2, NA)
  
  p1.1<- ggplot(data,aes(x=logFC, y=-log10(PValue)))+

    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "#999999")+

    geom_point(aes(color= -log10(PValue)), size = 3)+
    geom_text_repel(data = subset(top_genes, logFC > 0),
                    aes(label = gene_name, color = -log10(PValue)),
                    size = 5,
                    segment.size  = 0.8,
                    min.segment.length = 0, 
                    direction  = "y",
                    hjust = 0,
                    seed = 42,
                    box.padding = 0.4,
                    point.padding = 0.3,
                    na.rm = TRUE,
                    max.overlaps = 10
    ) +
    scale_color_gradientn(values = seq(0,1,0.2),
                          colors = c("#39489f","#39bbec","#f9ed36","#f38466","#b81f25"))+
    scale_size_continuous(
      range = c(2, 6),
      guide = "none"
    ) +
    scale_x_continuous(
      limits = c(-ceiling(max(abs(data$logFC))), ceiling(max(abs(data$logFC)))))+ # 刻度

    theme_bw() +
    theme(panel.grid = element_blank(),
          legend.position = "right" ,          
          legend.justification = c(0, 0.5),       
          legend.background = element_rect(    
            color = NA, fill = NA
          ),
          legend.key.width = unit(0.5, "cm"),  
          legend.key.height = unit(0.8, "cm"),   
          axis.title.x = element_text(size = 18), 
          axis.title.y = element_text(size = 18), 
          axis.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 16), 
          legend.text = element_text(size = 16),    
          legend.title = element_text(size = 16),  
          legend.title.align = 0.5
    )
  print(p1.1)
  sum(data$regulated == "up")
  sum(data$regulated == "down")
  ggsave(paste0('your path',do_seg,'_',condition[i,1],'-',condition[i,2],'.pdf'),p1.1, width = 8, height = 8)
}
write.csv(row.names(data)[which(data$PValue<0.05)], 'DEG.csv')