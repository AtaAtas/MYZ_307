# MYZ_307 HW_1
Homework and project files for the class MYZ 307: Machine Learning for Electrical and Electronics Engineering

In the code if the user wants to create new random uniform distribution data, they can change the csv folder path to distribution_path_random. This overwrites the distribution.csv file each time the code executes. If the user wants to observe an example distribution without overwriting it they can change the file path to distribution_path_fixed.

df = pd.read_csv(distribution_path_fixed) line does the file reading, don't change the df.to_csv(distribution_path_random, index=False) line.
