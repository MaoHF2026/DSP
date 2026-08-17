library(openxlsx)
library(readxl)
library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(pathview)
library(ggplot2)
library(GSEABase)
library(enrichplot)
library(tidyverse)

geneset <- list()
geneset[["hall"]] <- read.gmt("h.all.v2025.1.Hs.symbols.gmt")


gene_diff <- list()#target gene list

gsea_results <- list()
for (i in names(gene_diff)) {
  geneList <- gene_diff[[i]]$logFC
  names(geneList) <- toupper(rownames(gene_diff[[i]]))
  geneList <- sort(geneList, decreasing = TRUE)
  
  for (j in names(geneset)) {
    listnames <- paste(i, j, sep = "_")

    gsea_results[[listnames]] <- GSEA(
      geneList = geneList,
      TERM2GENE = geneset[[j]],
      verbose = FALSE,
      pvalueCutoff = 1,
      pAdjustMethod = "none",
      eps = 0
    )
  }
}


output_dir <- "GSEA_Results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Dir:", output_dir, "\n")
}

for (result_name in names(gsea_results)) {
  if (!is.null(gsea_results[[result_name]]) && 
      class(gsea_results[[result_name]])[1] == "gseaResult") {
    
    result_df <- as.data.frame(gsea_results[[result_name]])
    

    file_name <- gsub("[[:punct:]]+", "_", result_name)  
    file_name <- gsub("\\s+", "_", file_name)  
    

    file_path <- file.path(output_dir, paste0("gsea_", file_name, ".csv"))
    

    write.csv(result_df, 
              file = file_path,
              row.names = FALSE)
    
    cat("SAVED:", file_path, "\n")
  }
}



# List of result names (matching your GSEA results)
gsea_names <- c("ins_con_T2D_hall", "ins_con_OW_hall", "ins_OW_T2D_hall")

combined_results <- list()

# Extract NES and p-values from each GSEA result
for (result_name in gsea_names) {
  if (!is.null(gsea_results[[result_name]]) && class(gsea_results[[result_name]])[1] == "gseaResult") {
    result_df <- as.data.frame(gsea_results[[result_name]])
    
    nes_values <- result_df$NES
    p_values <- result_df$pvalue
    
    combined_results[[result_name]] <- data.frame(
      NES = nes_values,
      p_value = p_values
    )
  }
}


combined_df <- do.call(cbind, combined_results)


combined_df$Pathway <- rownames(result_df)

combined_df_heatmap <- combined_df[, -ncol(combined_df)] 
row.names(combined_df_heatmap) <- combined_df$Pathway
rownames(combined_df_heatmap) <- gsub("^HALLMARK_", "", rownames(combined_df_heatmap))

combined_df_heatmap_NES <- combined_df_heatmap[,c(1,3,5)]
combined_df_heatmap_P<- combined_df_heatmap[,c(2,4,6)]

stars <- ifelse(combined_df_heatmap_P < 0.05, "*", "")

pheatmap(
  combined_df_heatmap_NES,
  color = colorRampPalette(c("steelblue", "white", "#FFB3B3"))(100), 
  annotation_row_col = "Stars",  
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  show_rownames = TRUE,
  show_colnames = TRUE,
  cellwidth = 12,
  cellheight = 12,
  main = "GSEA Pathway Enrichment Heatmap",
  filename = "GSEA.pdf",
  display_numbers = stars,  
  fontsize_row = 8,  
  fontsize_number = 10  
)
dev.new()
