library(shiny)
library(tidyverse)

####data import

####VERY IMPORTANT: best practice is to have your .csv or other data file in the same directory as your .r file for the shiny app

setwd("~/Dropbox/data_viz_coursera_4/shiny_lecture_code")
dat<-read_csv("cel_volden_wiseman_coursera.csv")

dat <- dat%>%select(c(Congress=congress,Ideology=dwnom1,Party=dem))
dat$Party <- recode(dat$Party,`1`="Democrat",`0`="Republican")
dat=drop_na(dat)

####make the static figure for practice - this won't be displayed

ggplot(
  dat,
  aes(x=Ideology,color=Party,fill=Party))+
  geom_density(alpha=.5)+
  xlim(-1.5,1.5)+
  xlab("Ideology - Nominate Score")+
  ylab("Density")+
  scale_fill_manual(values=c("blue","red"))+
  scale_color_manual(values=c("blue","red"))
  
####Add facet wrap to see change over time
ggplot(
  dat,
  aes(x=Ideology,color=Party,fill=Party))+
  geom_density(alpha=.5)+
  xlim(-1.5,1.5)+
  xlab("Ideology - Nominate Score")+
  ylab("Density")+
  scale_fill_manual(values=c("blue","red"))+
  scale_color_manual(values=c("blue","red"))+
  facet_wrap(~Congress)

# Define UI for application that draws a figure
ui <- fluidPage(
  
  # Application title
  titlePanel("Ideology in Congress"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      sliderInput("my_cong",
                  "Congress:",
                  min = 93,
                  max = 114,
                  value = 93)
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("congress_distplot")
    )
  )
)

server <- function(input, output) {
  
  output$congress_distplot <- renderPlot({
    
    ggplot(
      filter(dat,Congress==input$my_cong),
      #####if the slider was set to the 93,  this filter line would look like:
      #filter(dat,Congress==93),
      #The value from the slider, assigned to "my_cong" in the UI goes into input$my_cong in the server function.
      
      aes(x=Ideology,color=Party,fill=Party))+
      geom_density(alpha=.5)+
      xlim(-1.5,1.5)+
      xlab("Ideology - Nominate Score")+
      ylab("Density")+
      scale_fill_manual(values=c("blue","red"))+
      scale_color_manual(values=c("blue","red"))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
