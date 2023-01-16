def get_lane_number(lane_name: str):
    lane_number = lane_name.split('_')[-1]
    return int(lane_number)
