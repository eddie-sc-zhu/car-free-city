import pandas as pd
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from agent import CommuterAgent  

def calc_avg_travel_time(model):
    total_time = 0
    count = 0
    for agent in model.schedule.agents:
        if agent.rides:
            total_time += agent.travel_time
            count += 1
    if count > 0:
        return total_time / count
    else:
        return 0

class BusModel(Model):
    def __init__(self,
                 data_path,
                 initial_revenue=0,
                 initial_services=1,
                 initial_convenience=1,
                 fare=2.5,
                 service_coeff=0.1,
                 convenience_effect=0.2,
                 travel_time_factor=0.05):
        """
        Args:
            data_path (file path): path to ridership csv data
            initial_revenue (float): starting revenue for Public Transit
            initial_services (float): starting amount of services for Public Transit
            initial_convenience (float): starting convenience for Public Transit
            fare (float): bus Fare (cost)
            service_coeff (float): multiplier for service
            convenience_effect (float): multiplier for convenience
            travel_time_factor (float): multiplier for travel time
        """
        # Read Data
        self.ridership_data = pd.read_csv(data_path)


        # Start Simulation Backbone
        self.current_day = 0  
        self.schedule = BaseScheduler(self)
        
        #### Public Transit Variables ####
        self.convenience = initial_convenience
        self.convenience_effect = convenience_effect

        self.services = initial_services
        self.service_coeff = service_coeff

        self.fare = fare
        self.revenue = initial_revenue
        self.ridership = 0  

        #### Negative Factors ####
        self.travel_time_factor = travel_time_factor
        
        self.datacollector = DataCollector(
            {
                "Ridership": lambda m: m.ridership,
                "Revenue": lambda m: m.revenue,
                "Services": lambda m: m.services,
                "Convenience": lambda m: m.convenience,
                "AvgTravelTime": lambda m: calc_avg_travel_time(m),
            }
        )
        self.datacollector.collect(self)

    def step(self):
        if self.current_day < len(self.ridership_data):
            daily_data = self.ridership_data.iloc[self.current_day]
            num_commuters_today = int(daily_data["Ridership"])
            self.schedule = BaseScheduler(self)
            for i in range(num_commuters_today):
                agent = CommuterAgent(i, self, min_convenience=0.15)
                self.schedule.add(agent)
            
            self.current_day += 1

        self.schedule.step()

        self.ridership = sum(1 for agent in self.schedule.agents if agent.rides)

        self.revenue += self.ridership * self.fare
        self.services += self.revenue * self.service_coeff

        avg_travel_time = calc_avg_travel_time(self)

        #   Convenience = (Services * convenience_effect) - (avg_travel_time * travel_time_factor)
        self.convenience = self.services * self.convenience_effect - avg_travel_time * self.travel_time_factor

        self.datacollector.collect(self)
