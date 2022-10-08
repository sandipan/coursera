library(shiny)
library(tidyverse)

#####create some data to interact with in the application

cities=c("Hong Kong","Macau","Dubai")

city1 <- tibble("city"=rep(cities[1],5),
                    "year"=seq(from=1990,to=1994,by=1),
                    #"unit"=letters[1:5],
                    "var1"=runif(5,0,5),
                    "var2"=runif(5,0,5),
                    "var3"=runif(5,0,5))
city2 <- tibble("city"=rep(cities[2],5),
                    "year"=seq(from=1990,to=1994,by=1),
                    #"unit"=letters[1:5],
                    "var1"=runif(5,0,5),
                    "var2"=runif(5,0,5),
                    "var3"=runif(5,0,5))
city3 <- tibble("city"=rep(cities[3],5),
                    "year"=seq(from=1990,to=1994,by=1),
                    #"unit"=letters[1:5],
                    "var1"=runif(5,0,5),
                    "var2"=runif(5,0,5),
                    "var3"=runif(5,0,5))

all_data <- bind_rows(city1,city2,city3)


#Skeleton UI

# ui<- fluidPage(
#   fluidRow(
#     column(4,
#            inputPanel(
#              sliderInput(...),
#              selectInput(...)
#            )),
#     column(4,
#            plotOutput(...)
#           ),
#     column(4,
#            tableOuput(...)
#            )
#   ),
#   fluidRow(
#     column(4,
#            inputPanel(
#              sliderInput(...),
#              selectInput(...)
#            )),
#     column(4,
#            plotOutput(...)
#     ),
#     column(4,
#            tableOuput(...)
#     )
#   )
#   
#  
# )




# Define UI
ui <- fluidPage(

  titlePanel("Multicolumn Format"),

  h2("These are the same inputs and figures repeated in two rows."),
  ####create multiple rows of input methods - rows are 12 units
  fluidRow(
        column(4,
       inputPanel(
           sliderInput(inputId="input_year",
                       label="Select Year",
                       value=c(1990,1994),
                       min=1990,
                       max=1994,
                       sep=""),

           selectInput(inputId="city1",
                       label="Which City Do you Want to Display?",
                       choices=c("Hong Kong","Macau","Dubai"),
                       selected=c("Hong Kong","Macau","Dubai")
                       )
        )
       ),

    column(4,
          plotOutput("plot1")
    ),

    column(4,
           tableOutput("table1")
    )
),

fluidRow(
  column(4,
         inputPanel(
           sliderInput(inputId="input_year2",
                       label="Select Year",
                       value=c(1990,1994),
                       min=1990,
                       max=1994,
                       sep=""),

           selectInput(inputId="city2",
                       label="Which City Do you Want to Display?",
                       choices=c("Hong Kong","Macau","Dubai"),
                       selected=c("Hong Kong","Macau","Dubai")
           )
         )
  ),

  column(4,
         plotOutput("plot2")
  ),

  column(4,
         tableOutput("table2")
  )
)
)

########write the server function that will generate the plots and tables to add to output

server <- function(input, output){

  output$table1 <- renderTable(

    {
    table_data1<-filter(
      all_data,
      city==input$city1 & year>=input$input_year[1] & year<=input$input_year[2])

    table_data1 <- table_data1%>%select(year,var1)
    table_data1$year <- as.integer(table_data1$year)
    table_data1

  }
    )

  output$plot1 <- renderPlot({

    plot_dat1<-filter(
      all_data,
      city==input$city1 & year>=input$input_year[1] & year<=input$input_year[2])

    ggplot(plot_dat1,
           aes(x=year,y=var1,group=city))+
      geom_line()

  })


  output$table2 <- renderTable(

    {
      table_data2<-filter(
        all_data,
        city==input$city2 & year>=input$input_year2[1] & year<=input$input_year2[2])

      table_data2 <- table_data2%>%select(year,var1)
      table_data2$year <- as.integer(table_data2$year)
      table_data2

    }
  )

  output$plot2 <- renderPlot({

    plot_dat2<-filter(
      all_data,
      city==input$city1 & year>=input$input_year2[1] & year<=input$input_year2[2])

    ggplot(plot_dat2,
           aes(x=year,y=var1,group=city))+
      geom_line()

  })
}

# Run the application
shinyApp(ui = ui, server = server)
