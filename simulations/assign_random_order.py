import csv
import pandas as pd
import random

sim_data_fn = "bias_type_sims/model_bias_comparison_sims_0.5_1.8.csv"
updated_data_fn = "bias_type_sims/model_bias_comparison_sims_0.5_1.8_randomgroups.csv"
min_seed = 1
max_seed = 3000
num_groups = 6 #Should be divisor of max_seed - min_seed + 1


#### Create seed assignments and groups by shuffling seed numbers ###
#Shuffle seeds
seed_list = list(range(min_seed, max_seed+1))
random.shuffle(seed_list)

#Assign groups, lined up with shuffled list
group_size = int((max_seed - min_seed + 1) / num_groups)
group_list = []
for group in range(1, num_groups+1):
    group_labels = [group] * group_size
    group_list = group_list + group_labels
shuffle_indices_groups = {"Seed": seed_list, "RandomGroup": group_list}

#Add as pandas column with left join, matching rows by seed
group_table = pd.DataFrame.from_dict(shuffle_indices_groups)
orig_table = pd.read_csv(sim_data_fn)

#Necessary because in an older version of the simulation code,
# re-running it duplicated seed runs in the output file!
# (it was appending instead of overwriting)
#There should be exactly num seeds x num models rows.
#If there are no duplicates (as there shouldn't be), this won't do anything
orig_table = orig_table.drop_duplicates()

updated_table = pd.merge(orig_table, group_table, how="left", on="Seed")

#Save updated table
updated_table.to_csv(updated_data_fn)







