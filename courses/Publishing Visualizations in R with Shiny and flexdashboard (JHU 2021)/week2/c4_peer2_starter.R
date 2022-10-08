library(shiny)
library(tidyverse)
library(plotly)
library(DT)

#####Import Data

dat <- read_csv(url("https://www.dropbox.com/s/uhfstf6g36ghxwp/cces_sample_coursera.csv?raw=1"))
dat <- dat %>% select(c("pid7","ideo5","newsint","gender","educ","CC18_308a","region"))
dat <- drop_na(dat)

#####Make your app

#####Hint: when you make the data table on page 3, you may need to adjust the height argument in the dataTableOutput function. Try a value of height=500

ui <- navbarPage(
  title="My Application",
  tabPanel("Page 1",
           sidebarLayout(
             sidebarPanel(
               sliderInput("ideo5",
                           "Select Five Point Ideology (1=Very liberal, 5=Very conservative)",
                           min = 1,
                           max = 5,
                           value = 3)
             ),
             mainPanel(
               tabsetPanel(
                 tabPanel("Tab 1", plotOutput("plot_tab1")), 
                 tabPanel("Tab 2", plotOutput("plot_tab2"))
               )
             )
           )
  ),
  tabPanel("Page 2",
           sidebarLayout(
             sidebarPanel(
               checkboxGroupInput("gender", "Select Gender",
                                 c(1:2))
             ),
             mainPanel(
               plotlyOutput("plot_reg")
             )
           )
        ),
  tabPanel("Page 3",
           sidebarLayout(
             sidebarPanel(
               selectInput("region", "Select Region", c(1:4), multiple=TRUE)
             ),
             mainPanel(
               DT::dataTableOutput("dat_region")
             )
           )
  )
)


server<-function(input,output){
  
  output$plot_tab1<-renderPlot(
    
    expr={
      #dat$pid7 <- as.factor(dat$pid7)
      dat %>% filter(ideo5==as.integer(input$ideo5)) %>% ggplot() + 
        geom_bar(aes(pid7)) + 
        #scale_x_discrete(breaks = seq(0, 8, by = 2),
        #                 labels = c('0', '2', '4', '6', '8')) + 
        scale_x_continuous(breaks = seq(0, 8, by = 2)) + 
        xlab('7 Point Party ID, 1=Very D, 7=Very R')
    }
    
  )
  output$plot_tab2<-renderPlot(
    
    expr={
      #dat$pid7 <- as.factor(dat$pid7)
      dat %>% filter(ideo5==as.integer(input$ideo5)) %>% ggplot() + 
        geom_bar(aes(CC18_308a)) + 
        #scale_x_discrete(breaks = seq(0, 8, by = 2),
        #                 labels = c('0', '2', '4', '6', '8')) + 
        scale_x_continuous(breaks = 1:4) + 
        xlab('Trump Support')
    }
    
  )
  output$plot_reg<-renderPlotly(
    
    expr={
      dat %>% filter(gender %in% input$gender) %>% 
        ggplot(aes(educ, pid7)) +       
        geom_jitter() +
        geom_point() + geom_smooth(method='lm')
      #ggplotly()
      #plot_ly(dat %>% filter(gender %in% input$gender), x=~educ, y=~pid7)
    }
    
  )
  output$dat_region <- DT::renderDataTable({
    dat %>% filter(region %in% input$region)
  })
  
}

shinyApp(ui=ui,server=server)

