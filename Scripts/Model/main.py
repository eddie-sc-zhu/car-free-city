import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np

def calc_avg_travel_time(model):
    '''if model.ridership > 0:
        travel_times = np.random.normal(loc=20, scale=5, size=model.ridership)
        return travel_times.mean()
    else:
        return 0'''
    return 20
    
class PublicTransitModel(Model):
    def __init__(self,
                 data_path="datasets/daily_ridership.csv",
                 initial_revenue=0,
                 initial_services=1,
                 initial_convenience=1,
                 intitial_employer=0,
                 fare=4.6,
                 service_coeff=0.1,
                 convenience_effect=0.2,
                 travel_time_factor=0.05,
                 awareness_coeff=0.05,           
                 employer_coeff=0.1,             
                 ineffectiveness_base=0.2,       
                 ineffectiveness_sensitivity=1.0 
                 ):
        
        # Read Data
        self.ridership_data = pd.read_csv(data_path)

        # Start Simulation Backbone
        self.current_day = 0

        #### Public Transit Variables ####
        self.convenience = initial_convenience
        self.convenience_effect = convenience_effect
        self.services = initial_services
        self.service_coeff = service_coeff
        self.fare = fare
        self.revenue = initial_revenue
        self.ridership = 0 

        #### Employer Variables ####
        self.employer_subsidy = intitial_employer

        #### Negative Factors ####
        self.travel_time_factor = travel_time_factor
        self.ineffectiveness_base = ineffectiveness_base   
        self.ineffectiveness_sensitivity = ineffectiveness_sensitivity  

        #### Positive Factors ####
        self.awareness_coeff = awareness_coeff           
        self.employer_coeff = employer_coeff      

        self.datacollector = DataCollector(
            {
                "Ridership": lambda m: m.ridership,
                "Revenue": lambda m: m.revenue,
                "Services": lambda m: m.services,
                "Convenience": lambda m: m.convenience,
                "AverageTravelTime": lambda m: calc_avg_travel_time(m),
            }
        )

        self.datacollector.collect(self)

    def step(self):
        if self.current_day < len(self.ridership_data):
            # remove this 
            print(self.current_day)

            # Read Baseline Data
            daily_data = self.ridership_data.iloc[self.current_day]
            num_commuters_today = int(daily_data["Ridership"])

            # Basically add upon the baseline data
            if self.current_day == 0:
                self.prev_baseline = num_commuters_today
                baseline_change = 0
            else:
                baseline_change = num_commuters_today - self.prev_baseline
                self.prev_baseline = num_commuters_today

            '''
            MODEL (REFERENCE):

            Loop 1: bus ridership (1)-> revenue (2)-> increased services (3)-> convenience (4)-> bus ridership
            Loop 2: convenience (5)-> awareness (6)-> employee transit passes (7)-> bus ridership

            Negative 1: bus ridership -> increased time for travel -> convenience 
            Negative 2: inneffectiveness -> bus ridership & awareness & employee interest
            '''

            # awareness -> inneffectiveness 
            ineffectiveness_penalty = min((self.ineffectiveness_base * self.ineffectiveness_sensitivity),1) 

            # Calculate new baseline from convenience (4)
            base_dynamic = num_commuters_today + baseline_change + (self.convenience * 5)

            # convenience * innefectiveness penalty -> awareness (5)
            awareness = self.convenience * self.awareness_coeff * (1 - ineffectiveness_penalty)

            # awareness -> employer     
            employer_effect = float(self.employer_subsidy)/base_dynamic + (awareness * (self.employer_coeff/10))           
            print(employer_effect)
            # Calculate Ridership
            dynamic_ridership = base_dynamic * (1 + employer_effect) * (1 - ineffectiveness_penalty)  
            self.ridership = dynamic_ridership
            self.current_day += 1
        else:
            pass

        # Ridership (* fare)-> Revenue
        self.revenue += self.ridership * self.fare

        # revenue -> services
        self.services += self.revenue * self.service_coeff

        
        avg_travel_time = calc_avg_travel_time(self)

        # Average (Services / Ridership) - Travel time -> convenience 
        # aggregate convenience equation = services * factor - average travel time * travel time factor
        self.convenience = (self.services * self.convenience_effect) / self.ridership - avg_travel_time * self.travel_time_factor
        self.datacollector.collect(self)
