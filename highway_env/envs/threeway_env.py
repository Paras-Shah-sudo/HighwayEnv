from typing import Optional, Tuple
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import Observation
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork

class ThreeWayEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "show_trajectories": True,
            "real_time_rendering": True
        })
        return config
    
    def define_spaces(self) -> None:
        return super().define_spaces()
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        return super().step(action)
    
    def render(self) -> Optional[np.ndarray]:
        return super().render()
    
    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=500):
        net = RoadNetwork()
        net.add_lane("a", "b", StraightLane([0, 0], [0, length], 3, \
                    line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)))
        net.add_lane("b", "c", StraightLane([0, length], [length, length], 3, \
                    line_types=(LineType.CONTINUOUS, LineType.NONE)))
        
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                        road.network.get_lane(("a", "b", 0)).position(0, 10), speed=30)
        self.road.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(2):
            v = vehicles_type(road,
                position=road.network.get_lane(("b", "a", 0))
                .position(200+100*i + 10*self.np_random.normal(), 0),
                heading=road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
                speed=20 + 5*self.np_random.normal(), enable_lane_change=False)
            v.target_lane_index = ("b", "c", 0)
            self.road.vehicles.append(v)