
line1_families = {} # insert line crresponding  dictionary here in the following format {"number": "letter"} eg : {
    # '3064':'G',
    # '3065':'G',
    # '3077':'G',
    # '3081':'G',
    # '3091':'G',
    # '7211':'S',
    # '7236':'S',
    # '7239':'S',
    # '7254':'S',
    # '7304':'B',
    # '7332':'S',}

line2_families = {
}

import pandas as pd
def get_alphabetical_family(batch_number,line = line1_families):
    new_batch_number = str(batch_number)

    if new_batch_number in line:
        return line[new_batch_number]
    else:
        raise KeyError(f"Batch number {new_batch_number} not found in line families.")
