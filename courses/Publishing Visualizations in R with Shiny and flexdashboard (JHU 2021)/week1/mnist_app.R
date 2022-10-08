#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)

#setwd('C:\\courses\\coursera\\R shiny\\week1')
train <- read.csv('mnist_sub.csv', stringsAsFactors = FALSE)
train$label <- as.factor(train$label)
tsne <- readRDS("tsne.rds")
pca <- readRDS("pca.rds")
mda_fit <- readRDS("mda.rds")
ae_fit <- readRDS("ae.rds")


# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Visualizing MNIST (handwritten digit) images"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            radioButtons("nsamples", "Number of samples:",
                         c("9",
                           "16",
                           "25")),
            selectInput("algo", "2D-Visualiation with embedding:",
                        c("pca" = "PCA",
                          "mds" = "MDS",
                          "autoencoder" = 'autoencoder',
                          "t-SNE" = "t-SNE"
                          ))
        ),
        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("showImages"),
           plotOutput("viz2D")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

    output$showImages <- renderPlot({
       n <- as.integer(input$nsamples)
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
               theme_bw() +
               theme(legend.position = "none", panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank())
           plist[[i]] <- p
       }
       do.call("grid.arrange", c(plist, ncol=as.integer(sqrt(n))))
    })
    
    output$viz2D <- renderPlot({
        algo <- input$algo
        if (algo == 'PCA') {
            pca_plot <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2], col = train$label)
            ggplot(pca_plot) + geom_text(aes(x=PC1, y=PC2, color=col, label=col), size=3, fontface = "bold") + 
                scale_color_manual(values = rainbow(10), breaks=1:10) + 
                theme_bw() + theme(legend.position = "none") +   
                theme(plot.title = element_text(hjust = 0.5)) +
                ggtitle('PCA plot') # + geom_point(aes(x=x, y=y, color=col))
        } else if (algo == 'MDS') {
            mda_plot <- data.frame(x = mda_fit$points[,1], y = mda_fit$points[,2], col = train$label)
            ggplot(mda_plot) + geom_text(aes(x=x, y=y, color=col, label=col), size=3, fontface = "bold") + 
                scale_color_manual(values = rainbow(10), breaks=1:10) + 
                theme_bw() + theme(legend.position = "none") +   
                theme(plot.title = element_text(hjust = 0.5)) +
                ggtitle('MDS plot') # + geom_point(aes(x=x, y=y, color=col))
        } else if (algo == 'autoencoder') {
            ae_plot <- data.frame(x = ae_fit[,1], y = ae_fit[,2], col = as.data.frame(train)$label)
            ggplot(ae_plot) + geom_text(aes(x=x, y=y, color=col, label=col), size=3, fontface = "bold") + 
                theme_bw() + theme(legend.position = "none") +   theme(plot.title = element_text(hjust = 0.5)) +
                ggtitle('AutoEncoder deepfeatures plot') # + geom_point(aes(x=x, y=y, color=col))
        } 
        else {
            tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2], col = train$label)
            ggplot(tsne_plot) + geom_text(aes(x=x, y=y, color=col, label=col), size=3, fontface = "bold") + 
                scale_color_manual(values = rainbow(10)) + 
                theme_bw() + theme(legend.position = "none") +   
                theme(plot.title = element_text(hjust = 0.5)) +
                ggtitle('t-SNE plot') # + geom_point(aes(x=x, y=y, color=col))
        }
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
