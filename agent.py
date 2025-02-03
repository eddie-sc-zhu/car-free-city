import random
from mesa import Agent

class CommuterAgent(Agent):
    def __init__(self, unique_id, model, min_convenience=5):
        """
        Args:
            unique_id (int): Agent_ID
            model (Model): Model Super Class
            min_convenience (float): Minimum Convenience Threshold (for agents to decide whether to take or not)
        """
        super().__init__(unique_id, model)
        self.min_convenience = min_convenience
        self.rides = False  
        self.travel_time = random.uniform(5, 30)

    def step(self):
        if self.model.convenience >= self.min_convenience:
            self.rides = True
        else:
            self.rides = False
