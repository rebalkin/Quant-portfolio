import numpy as np

def bird_on_a_wire(N_birds):

    birds_pos = np.zeros((N_birds+2, 1))  # Initialize positions of birds
    birds_pos[0] = -1000 #ficticious bird all the way to the left
    for i in range(1,N_birds+1):
         birds_pos[i] = np.random.uniform(0, 1)  # Randomly place birds on the wire
    birds_pos[N_birds+1] = 1000 #ficticious bird all the way to the right
    birds_pos = np.sort(birds_pos, axis=0) # Sort the positions of birds
    # print("Birds' positions:", birds_pos.flatten())
    paint_loc = np.zeros((N_birds+1, 1))  # Initialize paint locations
    for i in range(1,N_birds+1):
        if (birds_pos[i] - birds_pos[i-1]) < (birds_pos[i+1] - birds_pos[i]):
            paint_loc[i-1] = 1
        else:
            paint_loc[i] = 1
    # print("Paint locations:", paint_loc.flatten())
    paint =0
    for i in range(N_birds+1):
        if paint_loc[i] == 1:
            paint += birds_pos[i+1] - birds_pos[i]
    # print("Total paint used:", paint)      
    return paint[0]
