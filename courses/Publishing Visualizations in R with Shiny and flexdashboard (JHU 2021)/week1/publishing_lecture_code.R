library(shiny)
library(tidyverse)

#####DO NOT set your working directory in this app. This is will cause a problem when you try to upload the app online.

#setwd("~/Dropbox/data_viz_coursera_4/shiny_lecture_code")

dat<-read_csv("publish_practice.csv")

####this is just here for practice - it's what the final app is going to look like, as a static figure.
dat %>% ggplot(aes(x=varX,y=varY,color=Group))+geom_point()

# Define UI for application
ui <- fluidPage(
  
  #####Minimal sidebarLayout example:
  ####sidebarLayout(sidebarPanel(),mainPanel())
  
  sidebarLayout(
    sidebarPanel(
      checkboxGroupInput(
        inputId="checked_groups",
        label="Which groups do you want to display?",
        choices=c("a","b","c"),
        selected=c("a","b","c")
      )
    ),
      mainPanel(
        plotOutput("scatter")
      )
    )
  )


# Define server logic 
server <- function(input, output) {
  
  output$scatter<-renderPlot({
    
    plot_dat<-filter(dat, Group %in% input$checked_groups)
    
     ggplot(
       dat=plot_dat,
       aes(x=varX,y=varY,color=Group))+geom_point()
  }
  )
  
}


# Run the application 
shinyApp(ui = ui, server = server)



########This is the code used to generate the practice data
# 
#  varX<-seq(1:100)
#  varY<-varX+rnorm(100,10,10)
#  Group<-rep("a",100)
# 
#  groupa<-tibble(varX,varY,Group)
# 
#  varX<-seq(1:100)
#  varY<-rev(seq(1:100))+rnorm(100,10,10)
#  Group<-rep("b",100)
#  groupb<-tibble(varX,varY,Group)
# 
#  varX<-seq(1:100)
#  varY<-runif(100,0,100)
#  Group<-rep("c",100)
# 
#  groupc<-tibble(varX,varY,Group)
# 
#  
# publish_dat<-bind_rows(groupa,groupb,groupc)
#  
# write_csv(publish_dat,"publish_practice.csv")
