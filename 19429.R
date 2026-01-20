
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("GEOquery", "limma", "tidyverse"))

library(GEOquery)
library(limma)
library(tidyverse)

# 1. DOWNLOAD DATA
gse_id <- "GSE19429"
print(paste("Downloading", gse_id, "..."))

gse <- getGEO(gse_id, GSEMatrix = TRUE, AnnotGPL = TRUE)
gse <- gse[[1]]

# Extract Data
ex <- exprs(gse)
meta_raw <- pData(gse)

print(paste("Downloaded:", nrow(meta_raw), "Samples"))

# 2. CLEAN METADATA

meta_clean <- meta_raw %>%
  select(geo_accession, title, source_name_ch1) %>%
  mutate(
    # Create target column based on text search
    condition = case_when(
      grepl("healthy|control", title, ignore.case = TRUE) ~ "Healthy",
      grepl("MDS", title, ignore.case = TRUE) ~ "MDS",
      TRUE ~ "Unknown"
    )
  ) %>%
  filter(condition %in% c("Healthy", "MDS"))

# Align Data
common_samples <- intersect(rownames(meta_clean), colnames(ex))
ex <- ex[, common_samples]
meta_clean <- meta_clean[common_samples, ]

print("--- Class Balance (CRITICAL CHECK) ---")
print(table(meta_clean$condition))

# 3. DIFFERENTIAL EXPRESSION (Limma)
print("Running Limma...")
design <- model.matrix(~ 0 + factor(meta_clean$condition))
colnames(design) <- c("Healthy", "MDS")

# Contrast: MDS vs Healthy
fit <- lmFit(ex, design)
cont.matrix <- makeContrasts(Diff = MDS - Healthy, levels = design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2)

# 4. FILTERING
# We pick the top 50 genes with lowest p-value.
results <- topTable(fit2, adjust="fdr", number=Inf)

sig_genes <- results %>%
  filter(adj.P.Val < 0.01 & abs(logFC) > 1.0) %>%
  arrange(adj.P.Val) %>%
  head(50) # Keep small set for GA

# Handle Symbols
if("Gene.symbol" %in% colnames(results)){
  ids <- rownames(sig_genes)
  symbols <- sig_genes$Gene.symbol
  # Clean symbols
  symbols <- ifelse(symbols == "" | is.na(symbols), ids, symbols)
} else {
  ids <- rownames(sig_genes)
  symbols <- ids
}

print(paste("Selected", length(ids), "biomarkers."))

# 5. EXPORT
final_data <- ex[ids, ] %>% t() %>% as.data.frame()
colnames(final_data) <- symbols
final_data$target <- meta_clean$condition

write.csv(final_data, "GSE19429_Biomarker_Input.csv")
print("Saved: GSE19429_Biomarker_Input.csv")