# ==============================================================================
# 1. SETUP & LIBRARIES
# ==============================================================================
library(GEOquery)
library(limma)
library(Biobase)
library(ggplot2)
library(ggrepel)
library(dplyr)

# ==============================================================================
# 2. DATA LOADING & PRECISE FILTERING
# ==============================================================================
# Fetch dataset GSE58831
gset <- getGEO("GSE58831", GSEMatrix = TRUE, AnnotGPL = TRUE)
if (length(gset) > 1) idx <- grep("GPL570", attr(gset, "names")) else idx <- 1
eset <- gset[[idx]]

meta_data <- pData(eset)
target_col <- NULL

for (col_name in colnames(meta_data)) {
  if (any(grepl("disease status", meta_data[, col_name], ignore.case = TRUE))) {
    target_col <- col_name
    break
  }
}

if (is.null(target_col)) {
  stop("Error: Could not find a column containing 'disease status'. Check pData(eset).")
}

cat(paste("Found disease status info in column:", target_col, "\n"))
status_info <- meta_data[, target_col]

keep_indices <- c()
group_labels <- c()

for (i in 1:length(status_info)) {
  val <- status_info[i]
  
  if (grepl("disease status: Healthy control", val, fixed = TRUE)) {
    keep_indices <- c(keep_indices, i)
    group_labels <- c(group_labels, "Control")
    
  } else if (grepl("disease status: MDS", val, fixed = TRUE)) {
    keep_indices <- c(keep_indices, i)
    group_labels <- c(group_labels, "MDS")
  }
}

# Apply Filter
eset_clean <- eset[, keep_indices]
pData(eset_clean)$Group <- factor(group_labels, levels = c("Control", "MDS"))

# --- DATA CLEANING (Log & Probes) ---
ex <- exprs(eset_clean)
qx <- as.numeric(quantile(ex, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm=T))
LogC <- (qx[5] > 100) || (qx[6]-qx[1] > 50 && qx[2] > 0)
if (LogC) { 
  ex[which(ex <= 0)] <- NaN
  exprs(eset_clean) <- log2(ex) 
  cat("Log2 transformation applied.\n")
}

# Map Probes to Genes
gene_symbols <- fData(eset_clean)$`Gene symbol`
eset_final <- eset_clean[!is.na(gene_symbols) & gene_symbols != "", ]
exprs_data <- exprs(eset_final)
rownames(exprs_data) <- fData(eset_final)$`Gene symbol`

# Average duplicates
exprs_data <- avereps(exprs_data, ID = rownames(exprs_data))
group_list <- pData(eset_clean)$Group

print(paste("Data Ready: ", ncol(exprs_data), "Samples,", nrow(exprs_data), "Genes"))

# ==============================================================================
# 3. DIFFERENTIAL ANALYSIS (LIMMA)
# ==============================================================================
design <- model.matrix(~0 + group_list)
colnames(design) <- levels(group_list)

fit <- lmFit(exprs_data, design)
cont.matrix <- makeContrasts(Diff = MDS - Control, levels = design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2, 0.01)

results <- topTable(fit2, adjust="fdr", number=Inf)
results$Gene <- rownames(results)

# Thresholds
logFC_cutoff <- 0.5
pval_cutoff <- 0.05

results$Diff <- "NO"
results$Diff[results$logFC > logFC_cutoff & results$adj.P.Val < pval_cutoff] <- "UP"
results$Diff[results$logFC < -logFC_cutoff & results$adj.P.Val < pval_cutoff] <- "DOWN"
# ==============================================================================
# 3. 3D PCA PLOT 
# ==============================================================================

pca <- prcomp(t(exprs_data), scale. = TRUE)
pca_data <- as.data.frame(pca$x)
pca_data$Group <- group_list
pca_data$Sample <- rownames(pca_data)

# Calculate variance explained by each PC
var_explained <- round(100 * (pca$sdev^2 / sum(pca$sdev^2)), 1)

# Create Interactive 3D Plot
fig_3d <- plot_ly(pca_data, x = ~PC1, y = ~PC2, z = ~PC3, color = ~Group, 
                  colors = c('#2CA02C', '#D62728'), # Green=Control, Red=MDS
                  text = ~paste("Sample:", Sample),
                  type = "scatter3d", mode = "markers",
                  marker = list(size = 5, opacity = 0.8)) %>%
  layout(title = "3D PCA: GSE58831 (MDS vs Control)",
         scene = list(xaxis = list(title = paste0("PC1 (", var_explained[1], "%)")),
                      yaxis = list(title = paste0("PC2 (", var_explained[2], "%)")),
                      zaxis = list(title = paste0("PC3 (", var_explained[3], "%)"))))

print(fig_3d)

# ==============================================================================
# 4. VOLCANO PLOT
# ==============================================================================
volcano <- ggplot(results, aes(x = logFC, y = -log10(adj.P.Val), col = Diff)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = c("DOWN" = "dodgerblue3", "NO" = "gray80", "UP" = "firebrick3")) +
  geom_vline(xintercept = c(-logFC_cutoff, logFC_cutoff), linetype="dashed", alpha=0.5) +
  geom_hline(yintercept = -log10(pval_cutoff), linetype="dashed", alpha=0.5) +
  theme_minimal() +
  labs(title = "Volcano Plot: MDS vs Control (GSE58831)",
       subtitle = paste("Based on 'disease status' column | Up:", sum(results$Diff=="UP"), "Down:", sum(results$Diff=="DOWN")),
       x = "Log2 Fold Change", y = "-Log10 Adj. P-Value") +
  geom_text_repel(data = head(results, 10), aes(label = Gene), show.legend = FALSE)

print(volcano)
ggsave("GSE58831_Volcano.png", plot = volcano, width = 8, height = 6)

# ==============================================================================
# 5. EXPORT GA INPUT
# ==============================================================================
significant_genes <- results$Gene[results$adj.P.Val < 0.05]

if(length(significant_genes) < 50) {
  significant_genes <- results$Gene[1:500]
}

final_matrix <- t(exprs_data[significant_genes, ]) 
final_df <- as.data.frame(final_matrix)
final_df$Class_Label <- group_list

write.csv(final_df, "GA_Input_Data_GSE58831.csv", row.names = TRUE)