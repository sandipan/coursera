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

# 
# ui <- fluidPage(
#   
#   titlePanel("Tabsets"),
#   
#   sidebarLayout(
#     
#     sidebarPanel(
#       # Inputs excluded for brevity
#     ),
#     
#     mainPanel(
#       tabsetPanel(
#         tabPanel("Plot", plotOutput("plot")), 
#         tabPanel("Summary", verbatimTextOutput("summary")), 
#         tabPanel("Table", tableOutput("table"))
#       )
#     )
#   )
# )


ui <- fluidPage(
  
  titlePanel("Tabsets"),
  
  sidebarLayout(
      sidebarPanel(
      
      sliderInput(
        inputId="year_input",
        label="Year",
        min=1990,
        max=1994,
        value=1990,
        sep=""),
      
      selectInput(
        inputId="cities_input",
        label="City",
        choices=c("Hong Kong","Macau","Dubai"),
        selected=c("Hong Kong","Macau","Dubai"),
        multiple=TRUE
      )
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Variable 1", plotOutput("plot1")), 
        tabPanel("Variable 2", plotOutput("plot2"))
  )
)
)
)


server<-function(input,output){
  output$plot1<-renderPlot(
    
    expr={
    
    plot_dat1<-filter(all_data,
                      city %in% input$cities_input &
                        year==input$year_input)
    
    ggplot(plot_dat1,aes(x=city,y=var1,fill=city))+geom_bar(stat="identity")
  }
  
  )
  
  output$plot2<-renderPlot(
    
    expr={plot_dat2<-filter(all_data,
                      city %in% input$cities_input &
                      year==input$year_input)
    
          ggplot(plot_dat2,aes(x=city,y=var2,fill=city))+geom_bar(stat="identity")}
    
    )
  
  
}

shinyApp(ui=ui,server=server)

