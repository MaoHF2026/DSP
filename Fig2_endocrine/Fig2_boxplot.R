
library(readxl)
library(ggplot2)

data_all <- readxl::read_xlsx(
  "Your Path",
  sheet = "Table S2",
  skip = 2
)

data_all <- data_all[-24, ]

variables <- c(
  "Islet density (%)",
  "INS+ cells / Islet cells (%)",
  "GCG+ cells / Islet cells (%)"
)

colors <- c(
  "Control" = "#A8CAE8",
  "Overweight" = "#FFD966",
  "T2D" = "#E06666"
)

group_pairs <- list(
  c("Control", "Overweight"),
  c("Control", "T2D"),
  c("Overweight", "T2D")
)

all_wilcox_results <- list()

for (var in variables) {
  
  df <- data_all[, c("Group", var)]
  colnames(df) <- c("Group", "Value")
  df <- as.data.frame(df)
  
  df$Value <- trimws(as.character(df$Value))
  
  df$Value[
    df$Value %in% c(
      "N/A",
      "NA",
      "n/a",
      "na",
      ""
    )
  ] <- NA
  
  df$Value <- as.numeric(df$Value)
  
  df <- df[
    !is.na(df$Value),
  ]
  
  df$Group <- factor(
    df$Group,
    levels = c(
      "Control",
      "Overweight",
      "T2D"
    )
  )
  
  df <- df[
    !is.na(df$Group),
  ]
  
  wilcox_result <- do.call(
    rbind,
    lapply(
      group_pairs,
      function(x) {
        
        group1 <- df$Value[
          df$Group == x[1]
        ]
        
        group2 <- df$Value[
          df$Group == x[2]
        ]
        
        test <- wilcox.test(
          group1,
          group2,
          exact = FALSE
        )
        
        data.frame(
          Variable = var,
          Group1 = x[1],
          Group2 = x[2],
          n1 = length(group1),
          n2 = length(group2),
          W = unname(test$statistic),
          P_value = test$p.value
        )
      }
    )
  )
  
  wilcox_result$P_adj <- p.adjust(
    wilcox_result$P_value,
    method = "BH"
  )
  
  wilcox_result$Significance <- cut(
    wilcox_result$P_adj,
    breaks = c(
      -Inf,
      0.0001,
      0.001,
      0.01,
      0.05,
      Inf
    ),
    labels = c(
      "****",
      "***",
      "**",
      "*",
      "ns"
    )
  )
  
  all_wilcox_results[[var]] <- wilcox_result
  
  print(wilcox_result)
  
  y_min <- min(
    df$Value,
    na.rm = TRUE
  )
  
  y_max <- max(
    df$Value,
    na.rm = TRUE
  )
  
  y_range <- y_max - y_min
  
  if (y_range == 0) {
    y_range <- 1
  }
  
  lower_limit <- y_min - 0.08 * y_range
  upper_limit <- y_max + 0.12 * y_range
  
  p <- ggplot(
    df,
    aes(
      x = Group,
      y = Value,
      fill = Group
    )
  ) +
    
    geom_boxplot(
      outlier.shape = NA,
      alpha = 0.5,
      color = "black",
      linewidth = 0.5,
      width = 0.3
    ) +
    
    geom_jitter(
      fill = "black",
      shape = 21,
      size = 2,
      width = 0.2
    ) +
    
    scale_fill_manual(
      values = colors
    ) +
    
    scale_y_continuous(
      expand = expansion(
        mult = c(0, 0.05)
      )
    ) +
    
    coord_cartesian(
      ylim = c(
        lower_limit,
        upper_limit
      )
    ) +
    
    stat_summary(
      fun = max,
      geom = "errorbar",
      aes(
        ymax = after_stat(y),
        ymin = after_stat(y)
      ),
      width = 0.3,
      color = "black",
      linewidth = 0.5
    ) +
    
    stat_summary(
      fun = min,
      geom = "errorbar",
      aes(
        ymax = after_stat(y),
        ymin = after_stat(y)
      ),
      width = 0.3,
      color = "black",
      linewidth = 0.5
    ) +
    
    theme_minimal() +
    
    labs(
      title = "",
      x = "",
      y = var
    ) +
    
    theme(
      axis.line = element_line(
        color = "black",
        linewidth = 0.5
      ),
      axis.ticks = element_line(
        color = "black",
        linewidth = 0.5
      ),
      axis.text = element_text(
        size = 22,
        color = "black"
      ),
      axis.title = element_text(
        size = 22,
        color = "black"
      ),
      panel.grid = element_blank(),
      legend.position = "none",
      axis.title.y = element_text(
        margin = margin(
          t = 20,
          r = 0,
          b = 0,
          l = 0
        )
      )
    )
  
  print(p)
  
  file_name <- gsub(
    "[/\\\\:*?\"<>|%]",
    "_",
    var
  )
  
  ggsave(
    filename = paste0(
      "Your Path",
      file_name,
      ".pdf"
    ),
    plot = p,
    height = 6,
    width = 6
  )
}

all_wilcox_results_df <- do.call(
  rbind,
  all_wilcox_results
)

rownames(all_wilcox_results_df) <- NULL

print(all_wilcox_results_df)

