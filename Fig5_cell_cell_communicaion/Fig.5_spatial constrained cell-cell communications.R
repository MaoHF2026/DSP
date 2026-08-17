library(dplyr)
library(tidyr)
library(stringr)

expr_file <- "Path/Final_Aligned_Expression.csv"
meta_file <- "Path/Final_Aligned_Metadata.csv"
lr_file   <- "Path/LR_unique.csv" 

out_score_file <- "Path/CCC_Interaction_Scores_Detail_all.csv" 
out_stat_file  <- "Path/CCC_Statistics_Results_all.csv"        

message("Reading data ..")
expr <- read.csv(expr_file, row.names = 1, check.names = FALSE)
meta <- read.csv(meta_file)
lr_db <- read.csv(lr_file)

colnames(lr_db) <- tolower(colnames(lr_db)) 
available_genes <- rownames(expr)
valid_lr <- lr_db %>%
  filter(ligand %in% available_genes & receptor %in% available_genes)

message(paste("Primitive receptor pairs:", nrow(lr_db)))
message(paste("Effective receptor pairing (with gene expression):", nrow(valid_lr)))

if(nrow(valid_lr) == 0) stop("No receptor pairs expressed in the data were found! Please check the format of gene names.")


message("Building a cell specific matrix ..")

cell_types <- c("PeriMac", "IntraMac", "Alpha", "Beta")

cell_matrices <- list()

for (ct in cell_types) {
  samples <- meta$SampleID[meta$CellType == ct]
  
  if (length(samples) > 0) {
    mat <- t(expr[, samples])

    current_meta <- meta[match(samples, meta$SampleID), ]
    rownames(mat) <- current_meta$ROI_ID
    
    cell_matrices[[ct]] <- mat
  } else {
    warning(paste("Cell type not found in metadata:", ct))
  }
}


directions <- list(
  c(sender = "PeriMac",  receiver = "PeriMac"),
  c(sender = "PeriMac",  receiver = "IntraMac"),
  c(sender = "PeriMac",  receiver = "Alpha"),
  c(sender = "PeriMac",  receiver = "Beta"),

  c(sender = "IntraMac", receiver = "IntraMac"),
  c(sender = "IntraMac", receiver = "PeriMac"),
  c(sender = "IntraMac", receiver = "Alpha"),
  c(sender = "IntraMac", receiver = "Beta"),

  c(sender = "Alpha", receiver = "Alpha"),
  c(sender = "Alpha", receiver = "Beta"),
  c(sender = "Alpha", receiver = "PeriMac"),
  c(sender = "Alpha", receiver = "IntraMac"),
  
  c(sender = "Beta", receiver = "Beta"),
  c(sender = "Beta", receiver = "Alpha"),
  c(sender = "Beta", receiver = "PeriMac"),
  c(sender = "Beta", receiver = "IntraMac")
)

message("Start calculating communication strength ..")

results_list <- list()

for (pair in directions) {
  sender_type <- pair["sender"]
  receiver_type <- pair["receiver"]
  direction_name <- paste0(sender_type, "->", receiver_type)
  
  message(paste("  Calculating:", direction_name))
  
  mat_S <- cell_matrices[[sender_type]]
  mat_R <- cell_matrices[[receiver_type]]
  
  if (is.null(mat_S) | is.null(mat_R)) next
  
  common_rois <- intersect(rownames(mat_S), rownames(mat_R))
  
  if (length(common_rois) == 0) {
    message(paste("    Skip: No ROI contains both", sender_type, "and", receiver_type))
    next
  }
  
  sub_S <- mat_S[common_rois, , drop=FALSE]
  sub_R <- mat_R[common_rois, , drop=FALSE]
  
  roi_groups <- meta %>% 
    filter(ROI_ID %in% common_rois) %>%
    dplyr::select(ROI_ID, Group) %>%
    distinct() 
  
  roi_groups <- roi_groups[match(common_rois, roi_groups$ROI_ID), ]

  L_expr <- sub_S[, valid_lr$ligand, drop=FALSE]
  R_expr <- sub_R[, valid_lr$receptor, drop=FALSE]

  score_mat <- L_expr * R_expr
  colnames(score_mat) <- paste0(valid_lr$ligand, "_", valid_lr$receptor)

  score_df <- as.data.frame(score_mat)
  score_df$ROI_ID <- rownames(score_df)
  score_df$Group  <- roi_groups$Group
  score_df$Direction <- direction_name

  score_long <- score_df %>%
    pivot_longer(
      cols = -c(ROI_ID, Group, Direction),
      names_to = "LR_Pair",
      values_to = "Score"
    )
  
  results_list[[direction_name]] <- score_long
}

final_results <- do.call(rbind, results_list)

write.csv(final_results, out_score_file, row.names = FALSE)
message(paste("✅ Calculation completed! The detailed score has been saved to:", out_score_file))

message("Statistical testing is currently underway ..")

final_results$Group <- factor(final_results$Group, levels = c("Control", "Obesity", "T2D"))

stat_results <- final_results %>%
  group_by(Direction, LR_Pair) %>%
  summarise(
    Mean_Ctrl = mean(Score[Group == "Control"], na.rm=TRUE),
    Mean_Obesity = mean(Score[Group == "Obesity"], na.rm=TRUE),
    Mean_T2D = mean(Score[Group == "T2D"], na.rm=TRUE),
    
    P_Global_KW = tryCatch(kruskal.test(Score ~ Group)$p.value, error = function(e) NA),
    
    P_Obesity_vs_Ctrl = tryCatch(
      wilcox.test(Score[Group=="Obesity"], Score[Group=="Control"])$p.value, 
      error = function(e) NA
    ),
    
    P_T2D_vs_Ctrl = tryCatch(
      wilcox.test(Score[Group=="T2D"], Score[Group=="Control"])$p.value, 
      error = function(e) NA
    ),
    
    P_T2D_vs_Obesity = tryCatch(
      wilcox.test(Score[Group=="T2D"], Score[Group=="Obesity"])$p.value, 
      error = function(e) NA
    ),
    .groups = "drop"
  ) %>%

  mutate(
    Log2FC_Obesity_vs_Ctrl = log2((Mean_Obesity + 1e-9) / (Mean_Ctrl + 1e-9)),
    Log2FC_T2D_vs_Ctrl     = log2((Mean_T2D + 1e-9) / (Mean_Ctrl + 1e-9)),
    Log2FC_T2D_vs_Obesity  = log2((Mean_T2D + 1e-9) / (Mean_Obesity + 1e-9))
  ) %>%

  mutate(
    FDR_Global = p.adjust(P_Global_KW, method = "BH"),
    FDR_T2D_Ctrl = p.adjust(P_T2D_vs_Ctrl, method = "BH"),
    FDR_Obesity_Ctrl = p.adjust(P_Obesity_vs_Ctrl, method = "BH"),
    FDR_T2D_Obesity = p.adjust(P_T2D_vs_Obesity, method = "BH")
  ) %>%
  arrange(FDR_Global)

write.csv(stat_results, out_stat_file, row.names = FALSE)
message(paste("✅ Statistics completed! The statistical table has been saved to:", out_stat_file))

print("--- The top 5 most significant communication relationships in T2D vs Control ---")
top_hits <- stat_results %>% 
  arrange(P_T2D_vs_Ctrl) %>% 
  dplyr::select(Direction, LR_Pair, Mean_Ctrl, Mean_T2D, P_T2D_vs_Ctrl, Log2FC_T2D_vs_Ctrl) %>%
  head(5)
print(top_hits)
