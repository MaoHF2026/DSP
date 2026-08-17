library(factoextra)

rv <- genefilter::rowVars(exp)
select <- order(rv, decreasing = TRUE)
pca_data <- t(log10(exp[select,]+1))
expr_pca <- prcomp(pca_data,
                   scale = T,center = T)

fviz_screeplot(expr_pca, addlabels = TRUE, ylim = c(0, 80)) 

state_colors <- c("Control_intra" = "lightblue", "OW_intra" = "pink", "T2D_intra" = "red",
                  "Control_peri" = "darkblue", "OW_peri" = "darkorange", "T2D_peri" = "darkred")

PCA <- fviz_pca_ind(expr_pca,
                    label = 'none',
                    geom.ind = c('point','text'),
                    habillage = group,  
                    addEllipses = T,
                    ellipse.level = c(0.58),
                    palette = state_colors) +
  theme_bw((base_size=14))+  
  theme(text = element_text(size = 6),
        legend.margin = margin(-10),
        axis.text = element_text(size = 5, colour = 'black'),
        legend.text = element_text(size = 6),
        legend.title = element_blank(),
        legend.key.size = unit(0.5, "cm")
  )+
  ggtitle('')
PCA

