library(shiny)
library(tidyverse)

#####Import Data

dat<-read_csv(url("https://www.dropbox.com/s/uhfstf6g36ghxwp/cces_sample_coursera.csv?raw=1"))
dat<- dat %>% select(c("pid7","ideo5"))
dat<-drop_na(dat)

ui<- fluidPage(
  
  # Application title
  titlePanel(""),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      sliderInput("ideo5",
                  "Select Five Point Ideology (1=Very liberal, 5=Very conservative)",
                  min = 1,
                  max = 5,
                  value = 2)
    ),   
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("barPlot")
    )
  )
)

  

server<-function(input,output){
  output$barPlot <- renderPlot({
    dat %>% filter(ideo5==as.integer(input$ideo5)) %>% ggplot() + geom_bar(aes(pid7)) + xlab('7 Point Party ID, 1=Very D, 7=Very R')
  })
}

shinyApp(ui,server)
