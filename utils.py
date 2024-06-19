import os
import datetime

def path_to_save_file(sub_directory: str, file_name: str, include_time: bool = True):
    # Path to the current script
    current_script_path = os.path.abspath(__file__)

    # Path two levels up from the current script
    two_levels_up = os.path.dirname(os.path.dirname(current_script_path))

    # Path to the 'data' directory, two levels up from the script
    data_directory = os.path.join(two_levels_up, sub_directory)

    # Specify the full file path for where to write the file
    if include_time:
        # Get the current date and time without milliseconds
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")  # e.g., 2023-04-01_15-30-45
        return os.path.join(data_directory, f'{file_name}_{formatted_time}')
    return os.path.join(data_directory, file_name)