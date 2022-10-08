library(shiny)

questions <- function() {
  ##
  weekly_sales<-read.csv("sample_coursera.csv")
  
  ui<-fluidPage(
    titlePanel('Weekly Sales numbers'),
    tableOutput("table1")
  )
  
  server<-function(input,output){
    output$table1<-renderTable(weekly_sales)
  }
  
  shinyApp(ui,server)  
  
  ##
  ui<-fluidPage(
    column(6,
           textInput(inputId="text1",label="Some text",value="Test text"),
           textOutput("text1")
    ),
    column(6,
           numericInput(inputId="number1",label="A number",value="10"),
           textOutput("number1")
    )
  )
  
  server<-function(input,output){
    output$text1<-renderText(input$text1)
    output$number1<-renderText(input$number1)
  }
  shinyApp(ui,server)
  
  ##
  data(mtcars)
  
  ui<-fluidPage(
    sidebarLayout(
      sidebarPanel(
        selectInput(inputId="feature1","Feature",c("mpg","cyl"))),
      mainPanel(
        dataTableOutput("table1")))
    
  )
  
  server<-function(input,output){
    output$table1<-renderDataTable(mtcars[input$feature1])
  }
  
  shinyApp(ui,server)
  
  ##
  ui<-fluidPage(
    titlePanel("The Title"),    
    mainPanel(
      textInput(inputId = "text1",label="Type some letters",value=""),
      numericInput(inputId="number1",label="Type some numbers",value=""),
    ),
    tabsetPanel(
      tabPanel(title="Text", textOutput("tab_text")), 
      tabPanel(title="Numbers", textOutput("tab_number")))
    
  )
  
  server<-function(input,output){
    
    output$tab_text<-renderText(input$text1)
    output$tab_number<-renderText(input$number1)
    
  } 
  shinyApp(ui,server) 
  
  ##
  dat<-rnorm(1000,0,1)
  
  ui <- fluidPage(
    titlePanel("Sampling the Normal Distribution"),
    fluidRow(
      plotOutput("distPlot"),
      column(4,
             wellPanel(
               sliderInput("obs", "Number of observations:",  
                           min = 1, max = 1000, value = 500)))
    )
  )
  server<-function(input,output){
    output$distPlot<-renderPlot(hist(sample(dat,input$obs)))
  }
  
  shinyApp(ui,server)
  
  #my_dat<-read_csv("my_dat.csv")
  #ui<-fluidPage(tableOutput("table1"))
  #server<-function(input,output){
  
  
  dat<-rnorm(1000,0,1)
  
  ui <- fluidPage(
    
    titlePanel("Sampling the Normal Distribution"),
    sliderInput("obs", "Number of observations:",  
                min = 1, max = 1000, value = 500),
    plotOutput("distPlot"))
  
  
  server<-function(input,output){
    output$distPlot<-renderPlot(hist(sample(dat,input$obs)))
  }
  
  shinyApp(ui,server)
  
}




library(shiny)
library(plotly)

ui <- fluidPage(
  selectInput("choice", "Choose", choices = names(iris), selected = NULL),
  plotlyOutput("graph")
)

server <- function(input, output, session){
  
  output$graph <- renderPlotly({
    plot_ly(iris, x = ~get(input$choice), y = ~Sepal.Length, type = 'scatter', mode = 'markers')
  })
}

shinyApp(ui, server)


library(plotly)

fig <- plot_ly(mtcars, x = ~wt, y = ~hp, z = ~qsec,
               marker = list(color = ~mpg, colorscale = c('#FFE1A1', '#683531'), showscale = TRUE))
fig <- fig %>% add_markers()
fig <- fig %>% layout(scene = list(xaxis = list(title = 'Weight'),
                                   yaxis = list(title = 'Gross horsepower'),
                                   zaxis = list(title = '1/4 mile time')),
                      annotations = list(
                        x = 1.13,
                        y = 1.05,
                        text = 'Miles/(US) gallon',
                        xref = 'paper',
                        yref = 'paper',
                        showarrow = FALSE
                      )) %>%   add_markers(x = 2.8, y = 120, z = 20, color="red", marker=list(size=30,
                                                                                              opacity = .65,
                                                                                              line=list(width=2,
                                                                                                        color='black')))
fig

#plot_ly() %>% 
#  add_trace(data = mtcars,  x= ~wt, y=~hp, z=~qsec, type="mesh3d" ) 


library(plotly)

mtcars$am[which(mtcars$am == 0)] <- 'Automatic'
mtcars$am[which(mtcars$am == 1)] <- 'Manual'
mtcars$am <- as.factor(mtcars$am)

fig <- plot_ly(mtcars, x = ~wt, y = ~hp, z = ~qsec, color = ~am, colors = c('#BF382A', '#0C4B8E'))
fig <- fig %>% add_markers()
fig <- fig %>% layout(scene = list(xaxis = list(title = 'Weight'),
                                   yaxis = list(title = 'Gross horsepower'),
                                   zaxis = list(title = '1/4 mile time')))

fig


library(tidyverse)
library(plotly)
df <- data.frame(date = 
                   as.Date(c("01/01/1998", "10/01/1998", "15/01/1998", 
                             "25/01/1998", "01/02/1998", "12/02/1998", "20/02/1998"), "%d/%m/%Y"),
                 date2 = as.Date(c(NA, "10/01/1998", NA, 
                                   NA, NA, NA, NA), "%d/%m/%Y"),
                 counts = c(12, 10, 2, 24, 15, 1, 14),
                 yes_no = c("yes", "yes", "yes", "no", "no", "no", "no"))

gg <- ggplot(df, aes(x = date, y = counts)) +
  geom_line() +
  geom_ribbon(aes(ymin = 0, ymax = counts), color = NA, alpha = 0.5) + aes(fill = yes_no) + 
  scale_fill_brewer(name = "status 1", palette = "Accent") +
  geom_vline(data=df, mapping = aes(xintercept = as.numeric(date2), col = 'mystatistic')) +
  scale_color_manual(name = "statistics", values = c("mystatistic" = "red")) 
gg
ggplotly(gg)

gg_1 <-  ggplot(df, aes(x = date, y = counts)) +
  geom_line() +
  geom_ribbon(aes(ymin = 0, ymax = counts), color = NA, alpha = 0.5) + aes(fill = yes_no) + 
  scale_fill_brewer(name = "status 1", palette = "Accent")
gg_1 <- ggplotly(gg_1)
gg_2 <- gg_1 + geom_vline(aes(xintercept = as.numeric(date2))) + aes(col = 'mystatistic') +
  scale_color_manual(name = "statistics", values = c("mystatistic" = "red")) 
gg_2 <- ggplotly(gg_2)

gg <- ggplotly(gg) 
n = length(unique(df$z))
gg1 <- gg %>% style(gg, showlegend = FALSE, traces = 1:2) 
ggplotly(gg1)

gg <- ggplot(df, aes(x = date, y = counts)) +
  geom_line() +
  geom_ribbon(aes(ymin = 0, ymax = counts), color = NA, alpha = 0.5) + aes(fill = factor(yes_no)) +
  scale_fill_brewer(name = "status 1", palette = "Accent") +
  geom_vline(data=df, mapping = aes(xintercept = as.numeric(date2))) + #, col = 'mystatistic')) +
  #guides(color=guide_legend("mystatistics"))
  aes(col = 'mystatistic') +
  scale_color_manual(name = "statistics", values = c("mystatistic" = "red")) 
gg
ggplotly(gg)



df <- data.frame( x = 1:8, y=1:16, z = LETTERS[1:4])
q <- ggplot(df, aes(x = x, y = y, group=z)) + geom_point(aes(shape=z)) + geom_line()  + aes(color=z)
p <- ggplotly(q)
qq <- p %>% style(p, showlegend = FALSE, traces = 5:8)


df <- data.frame( x = 1:8, y=1:16, z = LETTERS[1:4])
p <- ggplot(df, aes(x = x, y = y)) + geom_point(aes(color=factor(z)), show.legend=FALSE)
p1 <- p + geom_line(aes(color=factor(z), linetype=factor(z))) 
p2 <- p + geom_line(aes(color=factor(z) )) + aes(linetype=factor(z))
p3<- ggplotly(p1) 
n = length(unique(df$z))
p4  <- p3 %>% style(p3, showlegend = FALSE, traces = 1:n) 

