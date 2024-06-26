library(monocle3)
library(Matrix)
library(Seurat)

monocle_v3 <- function(seed, train, cellmeta_dir, genemeta_dir, save_seurat, save_dir, counts_dir, barcode_dir) {
    set.seed(seed)
    counts <- assay(train,'X')
    cell_metadata = read.csv(cellmeta_dir, encoding='utf-8')
    gene_metadata = read.csv(genemeta_dir, encoding='utf-8')

    colnames(counts) <- cell_metadata[["Barcode"]]
    rownames(counts) <- gene_metadata[["gene"]]

    seurat <- CreateSeuratObject(counts=counts)
    seurat@meta.data <- cbind(cell_metadata, seurat@meta.data)
    rownames(seurat@meta.data) <- colnames(seurat)

    seurat <- FindVariableFeatures(seurat, selection.method = "vst", nfeatures = 2000)
    seurat <- ScaleData(seurat)
    seurat <- RunPCA(seurat, features = VariableFeatures(object = seurat), npcs=30, verbose=FALSE)
    seurat <- RunUMAP(seurat, dims = 1:10)
    runned_umap <- seurat@meta.data[, c('UMAP1', 'UMAP2')]
    colnames(runned_umap) <- c('UMAP_1', 'UMAP_2')
    seurat@reductions$umap@cell.embeddings <- as.matrix(runned_umap)

    if (save_seurat == TRUE) {
        saveRDS(seurat, file.path(save_dir,'seo_annotated.rds'))
    }

    gene_annotation <- as.data.frame(rownames(seurat@reductions$pca@feature.loadings), row.names = rownames(seurat@reductions$pca@feature.loadings))
    colnames(gene_annotation) <- "gene_short_name"

    ### cell_metadata
    cell_metadata <- as.data.frame(seurat@assays[["RNA"]]@counts@Dimnames[[2]], row.names = seurat@assays[["RNA"]]@counts@Dimnames[[2]])
    colnames(cell_metadata) <- "barcode"

    ### expression matrix
    New_matrix <- seurat@assays[["RNA"]]@counts
    New_matrix <- New_matrix[rownames(seurat@reductions[["pca"]]@feature.loadings), ]
    expression_matrix <- New_matrix

    cds <- new_cell_data_set(expression_matrix, cell_metadata = cell_metadata, gene_metadata = gene_annotation)

    cds <- preprocess_cds(cds, num_dim = 50)
    cds <- reduce_dimension(cds, reduction_method="UMAP")

    recreate.partition <- c(rep(1, length(cds@colData@rownames)))
    names(recreate.partition) <- cds@colData@rownames
    recreate.partition <- as.factor(recreate.partition)

    cds@clusters@listData[["UMAP"]][["partitions"]] <- recreate.partition
    cds@int_colData@listData[["reducedDims"]][["UMAP"]] <-seurat@reductions[["umap"]]@cell.embeddings

    cds <- cluster_cells(cds, cluster_method="louvain")
    cds <- learn_graph(cds)

    png(file = "cluster.png", width = 800, height = 600, res = 300)
    plot_cells(cds, color_cells_by = "partition")
    dev.off()

    cds <- order_cells(cds, reduction_method = "UMAP")
    
    png(file = "pseudotime.png", width = 800, height = 600, res = 300)
    plot_cells(cds,
               color_cells_by = "pseudotime",
               label_cell_groups=FALSE,
               label_leaves=FALSE,
               label_branch_points=FALSE,
               graph_label_size=1.5)
    dev.off()
}

save_dir <- "/volume1/home/kliu/Data/temp"
train <- readRDS(paste(save_dir, "/sce_data.rds", sep=""))
seed = 42
save_seurat <- FALSE
counts_dir <- paste(save_dir, "/counts.mtx", sep="")
barcode_dir <- paste(save_dir, "/barcode.csv", sep="")
cellmeta_dir <- paste(save_dir, "/cell_metadata.csv", sep="")
genemeta_dir <- paste(save_dir, "/gene_metadata.csv", sep="")

monocle_v3(seed, train, cellmeta_dir, genemeta_dir, save_seurat, save_dir, counts_dir, barcode_dir)



    colnames(counts) <- cell_metadata[["Barcode"]]
    rownames(counts) <- gene_metadata[["gene"]]

    seurat <- CreateSeuratObject(counts=counts)
    seurat@meta.data <- cbind(cell_metadata, seurat@meta.data)
    rownames(seurat@meta.data) <- colnames(seurat)

    seurat <- FindVariableFeatures(seurat, selection.method = "vst", nfeatures = 2000)
    seurat <- ScaleData(seurat)
    seurat <- RunPCA(seurat, features = VariableFeatures(object = seurat), npcs=30, verbose=FALSE)
    seurat <- RunUMAP(seurat, dims = 1:10)
    runned_umap <- seurat@meta.data[, c('UMAP1', 'UMAP2')]
    colnames(runned_umap) <- c('UMAP_1', 'UMAP_2')
    seurat@reductions$umap@cell.embeddings <- as.matrix(runned_umap)

    if (save_seurat == TRUE) {
        saveRDS(seurat, file.path(save_dir,'seo_annotated.rds'))
    }

    gene_annotation <- as.data.frame(rownames(seurat@reductions$pca@feature.loadings), row.names = rownames(seurat@reductions$pca@feature.loadings))
    colnames(gene_annotation) <- "gene_short_name"

    ### cell_metadata
    cell_metadata <- as.data.frame(seurat@assays[["RNA"]]@counts@Dimnames[[2]], row.names = seurat@assays[["RNA"]]@counts@Dimnames[[2]])
    colnames(cell_metadata) <- "barcode"

    ### expression matrix
    New_matrix <- seurat@assays[["RNA"]]@counts
    New_matrix <- New_matrix[rownames(seurat@reductions[["pca"]]@feature.loadings), ]
    expression_matrix <- New_matrix

    cds <- new_cell_data_set(expression_matrix, cell_metadata = cell_metadata, gene_metadata = gene_annotation)

    cds <- preprocess_cds(cds, num_dim = 50)
    cds <- reduce_dimension(cds, reduction_method="UMAP")

    recreate.partition <- c(rep(1, length(cds@colData@rownames)))
    names(recreate.partition) <- cds@colData@rownames
    recreate.partition <- as.factor(recreate.partition)

    cds@clusters@listData[["UMAP"]][["partitions"]] <- recreate.partition
    cds@int_colData@listData[["reducedDims"]][["UMAP"]] <-seurat@reductions[["umap"]]@cell.embeddings

    cds <- cluster_cells(cds, cluster_method="louvain")
    cds <- learn_graph(cds)
    monocle3::plot_cells(cds, color_cells_by = "partition")
    cds <- order_cells(cds, reduction_method = "UMAP")

    monocle3::plot_cells(cds,
                color_cells_by = "pseudotime",
                label_cell_groups=FALSE,
                label_leaves=FALSE,
                label_branch_points=FALSE,
                graph_label_size=1.5)
}

save_dir <- "/volume1/home/kliu/Data/temp"
train <- readRDS(paste(save_dir, "/saved_data.rds", sep=""))
seed = 42
save_seurat <- TRUE
counts_dir <- paste(save_dir, "/counts.mtx", sep="")
barcode_dir <- paste(save_dir, "/barcode.csv", sep="")
cellmeta_dir <- paste(save_dir, "/cell_metadata.csv", sep="")
genemeta_dir <- paste(save_dir, "/gene_metadata.csv", sep="")

monocle_v3(seed, train, cellmeta_dir, genemeta_dir, save_seurat, save_dir, counts_dir, barcode_dir)
