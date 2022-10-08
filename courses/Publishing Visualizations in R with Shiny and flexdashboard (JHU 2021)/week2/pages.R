library(shiny)
library(tidyverse)


cities=c("Hong Kong","Macau","Dubai")

city1 <- data.frame("city"=rep(cities[1],5),
                    "year"=seq(from=1990,to=1994,by=1),
                    #"unit"=letters[1:5],
                    "var1"=runif(5,0,5),
                    "var2"=runif(5,0,5),
                    "var3"=runif(5,0,5))
city2 <- data.frame("city"=rep(cities[2],5),
                    "year"=seq(from=1990,to=1994,by=1),
                    #"unit"=letters[1:5],
                    "var1"=runif(5,0,5),
                    "var2"=runif(5,0,5),
                    "var3"=runif(5,0,5))
city3 <- data.frame("city"=rep(cities[3],5),
                    "year"=seq(from=1990,to=1994,by=1),
                    #"unit"=letters[1:5],
                    "var1"=runif(5,0,5),
                    "var2"=runif(5,0,5),
                    "var3"=runif(5,0,5))

all_data <- bind_rows(city1,city2,city3)


# library(shiny)
# 
# ui <- navbarPage("My Application",
#                  tabPanel("Component 1"),
#                  tabPanel("Component 2"),
#                  tabPanel("Component 3")
# )
# 
# server<-function(input,output){}
# 
# shinyApp(ui=ui,server=server)





ui <- navbarPage(
                 title="My Application",
                 tabPanel("Component 1",
                           sliderInput(
                            inputId="year_input",
                            label="Year",
                            min=1990,
                            max=1994,
                            value=1990,
                            sep=""),
                           plotOutput("plot1")
                          ),
                 tabPanel("Component 2"),
                 tabPanel("Component 3")
)

server<-function(input,output){
  output$plot1<-renderPlot({
                  
                  plot_dat<-filter(all_data,year==input$year_input)
    
                  ggplot(plot_dat,aes(x=city,y=var1,fill=city))+geom_bar(stat="identity")
  })
  
  
}

shinyApp(ui=ui,server=server)

