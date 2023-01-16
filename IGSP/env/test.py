from offloading import Offloading
import numpy as np
from vehicle import Vehicle
if __name__ == '__main__':
    off = Offloading()
    off.reset()

    # for vehicle in off.vehicles:
    #     print("vehicle{} location:".format(vehicle.id), vehicle.get_location)
    service_vehicle = Vehicle(id=3, loc_x=200, loc_y=1, direction=1, velocity=15)
    service_vehicle.create_work()

    vehicles = off.vehicles

    # off.distribute_task()
    # off.distribute_resource()
    # off.compute_rate(service_vehicle, vehicles[0])
    off.compute_persist(service_vehicle, vehicles[1])
