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


# ui <- fluidPage(
#   
#   titlePanel("Application Title"),
#   
#   navlistPanel(
#     "Header A",
#     tabPanel(title="Component 1",
#             sliderInput(),
#             checkboxGroupInput(),
#             plotOutput()
#             ),
#     tabPanel("Component 2"),
#     "Header B",
#     tabPanel("Component 3"),
#     tabPanel("Component 4"),
#     "-----",
#     tabPanel("Component 5")
#   )
# )



library(shiny)

ui <- fluidPage(
  
  titlePanel("Application Title"),
  
  navlistPanel(
    "Header A",
    tabPanel(
      title="Component 1",
      sliderInput(
        inputId="year_input",
        label="Year",
        min=1990,
        max=1994,
        value=1990,
        sep=""),
      plotOutput("plot1")
    ),
   
     tabPanel("Component 2",
              sidebarLayout(
                position="right",
                sidebarPanel(
                    selectInput(
                       inputId="cities_input",
                       label="City",
                       choices=c("Hong Kong","Macau","Dubai"),
                       selected=c("Hong Kong","Macau","Dubai"),
                       multiple=TRUE
                       )
                    ),
                    mainPanel(
                      plotOutput("plot2"))
     )
     ),
           
    "Header B",
    tabPanel("Component 3"),
    tabPanel("Component 4")
  )
)



server<-function(input,output){
  output$plot1<-renderPlot({
    
    plot_dat1<-filter(all_data,year==input$year_input)
    
    ggplot(plot_dat1,aes(x=city,y=var1,fill=city))+geom_bar(stat="identity")
  })
  
  output$plot2<-renderPlot({
    
    plot_dat2<-filter(all_data,city %in% input$cities_input)
    
    ggplot(plot_dat2,aes(x=city,y=var1,fill=city))+geom_bar(stat="identity")
  })
  
  
}

shinyApp(ui=ui,server=server)

