## calling the installed package
train<- read.csv(file.choose()) ## Choose the train.csv file downloaded from the link above  

#train <- do.call(rbind, lapply(split(train, train$label), function(x) x[sample(1:nrow(x), 500),]))
#write.csv(train, 'mnist_sub.csv', row.names = FALSE)
#c('#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8')) 
library(Rtsne)
## Curating the database for analysis with both t-SNE and PCA
Labels<-train$label
train$label<-as.factor(train$label)
## for plotting
colors = rainbow(length(unique(train$label)))
names(colors) = unique(train$label)

## Executing the algorithm on curated data
tsne <- Rtsne(train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
saveRDS(tsne, "tsne.rds")
tsne <- readRDS("tsne.rds")

pca_res <- prcomp(train[,-1]) #, scale. = TRUE
saveRDS(pca_res, "pca.rds")
pca <- readRDS("pca.rds")

library(ggfortify)
autoplot(pca, data = train, colour = 'label', label = TRUE, label.size = 3)

#dat %>% filter(pid7==input$pid7) %>% ggplot() + geom_bar(aes(ideo5, pid7), stat='count')

d <- dist(train[,-1]) # euclidean distances between the rows
mda_fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
saveRDS(mda_fit, "mda.rds")
mda_fit <- readRDS("mda.rds")

library(h2o)
h2o.init(nthreads=-1)
train <- h2o.importFile(file.choose(), header=TRUE)
ae <- h2o.deeplearning(x = 1:784,
                             training_frame = train[,-1],
                             autoencoder = TRUE,
                             epochs = 300,   ## I'm needing about 140 to 150 to hit early stopping
                             
                             model_id = "ae",
                             
                             train_samples_per_iteration = nrow(train),
                             score_interval = 0,
                             score_duty_cycle = 1.0,
                             
                             hidden = c(100,2,100),
                             activation = "Tanh"
)
ae_fit <- h2o.deepfeatures(ae, train[,-1], 2)
ae_fit <- as.data.frame(ae_fit)
saveRDS(ae_fit, "ae.rds")
ae_fit <- readRDS("ae.rds")


#exeTimeTsne<- system.time(Rtsne(train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500))

## Plotting
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=train$label, col=colors[train$label])

train$label <- as.factor(train$label)

library(ggplot2)
tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2], col = train$label)
ggplot(tsne_plot) + geom_text(aes(x=x, y=y, color=col, label=col), size=3, fontface = "bold") + 
  theme_bw() + theme(legend.position = "none") +   theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle('t-SNE plot') # + geom_point(aes(x=x, y=y, color=col))

pca_plot <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2], col = train$label)
ggplot(pca_plot) + geom_text(aes(x=PC1, y=PC2, color=col, label=col), size=3, fontface = "bold") + 
  theme_bw() + theme(legend.position = "none") +   theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle('PCA plot') # + geom_point(aes(x=x, y=y, color=col))

mda_plot <- data.frame(x = mda_fit$points[,1], y = mda_fit$points[,2], col = train$label)
ggplot(mda_plot) + geom_text(aes(x=x, y=y, color=col, label=col), size=3, fontface = "bold") + 
  theme_bw() + theme(legend.position = "none") +   theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle('MDS plot') # + geom_point(aes(x=x, y=y, color=col))

ae_plot <- data.frame(x = ae_fit[,1], y = ae_fit[,2], col = as.data.frame(train)$label)
ggplot(ae_plot) + geom_text(aes(x=x, y=y, color=col, label=col), size=3, fontface = "bold") + 
  theme_bw() + theme(legend.position = "none") +   theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle('AutoEncoder features plot') # + geom_point(aes(x=x, y=y, color=col))

library(gridExtra)
n <- 16
samples <- train[sample(1:nrow(train), n),]
plist <- list()
for (i in 1:n) {
  m <- matrix(as.integer(samples[i,-1]), nrow=28, byrow=T)
  m <- t(apply(m, 2, rev))
  rownames(m) <- colnames(m) <- 1:28
  df <- as.data.frame(m) %>% gather(key='y', value='pixel')
  df$y <- as.integer(df$y)
  df$x <- rep(1:28, 28)
  p <- ggplot(df, aes(x, y, fill= pixel)) + 
    geom_tile() +
    scale_fill_gradient(low = "black", high = "white") + 
    labs(x = NULL, y = NULL) + 
    guides(x = "none", y = "none") +
    theme(legend.position = "none")
  plist[[i]] <- p
}
#theme_void() + 
#scale_fill_manual(values=grey.colors(256))


#for (i in 1:28) {m[,i] <- as.integer(m[,i])}
#plist <- list()
#plist[[1]] <- image(m, col=gray.colors(255))

do.call("grid.arrange", c(plist, ncol=as.integer(sqrt(n))))

library(png)
library(grid)
library(gridExtra)

library(RCurl)
library(jpeg)
library(grid)
library(gridExtra)
library(ggplot2)

urls <- c('https://i.imgur.com/r4PT8WQ.jpg', 'https://i.imgur.com/b3Vv2nq.jpg', 'https://i.imgur.com/oT4qU.jpg', 'https://i.imgur.com/SDUmg.jpg')
imgs <- list()
for (i in 1:length(urls)) {
  imgs[[i]] <- readJPEG(getURLContent(urls[i]))
}

library(RCurl)
library(png)
library(grid)
library(gridExtra)
library(ggplot2)
urls <- c('https://i.imgur.com/TEbkTqu.png', 'https://i.imgur.com/tnsjMFJ.png', 'https://i.imgur.com/VUZgJBs.png', 'https://i.imgur.com/FZ28d3w.png')
imgs <- list()
for (i in 1:length(urls)) {
  imgs[[i]] <- readPNG(getURLContent(urls[i])) #readJPEG
}

plist <- list()
for (i in 1:length(imgs)) {
  plist[[i]] <- ggplot() +
    annotation_custom(rasterGrob(imgs[[i]], interpolate=TRUE), xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)  +
    labs(x = NULL, y = NULL) + 
    guides(x = "none", y = "none") +
    theme_bw() +
    theme(legend.position = "none", panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
}
#do.call("grid.arrange", c(plist, ncol=2))
marrangeGrob(plist, nrow=2, ncol=2, respect=TRUE)




library(microbenchmark)

m <- 10
n <- 1000
A <- matrix(rnorm(m*n), m, n)
AtA <- t(A) %*% A
dim(AtA)
qr(AtA)$rank
AAt <- A %*% t(A)
dim(AAt)
qr(AAt)$rank

mbm <- microbenchmark("PCA_AtA" = { 
    res <- eigen(AtA)
    val1 <- res$values
    vec1 <- res$vectors
  },
  "PCA_AAt" = {
    res <- eigen(AAt)
    val2 <- res$values
    vec2 <- res$vectors
    vec2 <- t(A) %*% vec2
  })
mbm

library(ggplot2)
autoplot(mbm)

length(val1)
length(val2)
val1 <- round(val1, 3)
val2 <- round(val2, 3)
val1
val2

round(val1)
all.equal(val1[1:10], val2)

sqrt(sum(vec1[,1]^2))

sqrt(sum(vec2[,1]^2))

vec2 <- apply(vec2, 2, function(x) x / sqrt(sum(x^2)))

sqrt(sum(vec2[,1]^2))
