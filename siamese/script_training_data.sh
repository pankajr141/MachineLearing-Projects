python3 create_training_data.py "/data/type1" /trainingdata 0 100
python3 create_training_data.py "/data/type2" /trainingdata 0 100
python3 create_training_data.py "/data/type3" /trainingdata 0 100
python3 create_training_data.py "/data/type4" /trainingdata 0 100
python3 create_training_data.py "/data/type5" /trainingdata 0 100
python3 create_training_data.py "/data/type6" /trainingdata 0 100
python3 generate_combination_file.py /trainingdata 1000000
