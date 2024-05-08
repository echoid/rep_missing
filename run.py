import os

data_list = ["banknote", "climate_model_crashes", "connectionist_bench_sonar", "qsar_biodegradation", "yeast"]
#data_list = [ "connectionist_bench_sonar", "qsar_biodegradation", "yeast"]
#data_list = ['concrete_compression','wine_quality_red','wine_quality_white','yacht_hydrodynamics']
type_list = ["logistic", "mar", "mcar"]
para_list = [0.5]
full_norm = ["full"]


# data_list = ["banknote"]
# #data_list = ['concrete_compression','wine_quality_red','wine_quality_white','yacht_hydrodynamics']
# type_list = ["mcar"]
# para_list = [0.3]
# full_norm = ["full"]

command = f"git add ."
print(command)
os.system(command)


# for data in data_list:
#     for types in type_list:
#         for para in para_list:
#             for f_n in full_norm:
#             #command = f"python main_demo.py ./{data}_{types}/"
#                 #command = f"python baseline.py {data} {types} {para} {f_n}"
#                 #command = f"python ppca_run.py {data} {types} {para} {f_n}"
#                 #command = f"python kernal_pca_run.py {data} {types} {para} {f_n}"
#                 command = f"python ik_run.py {data} {types} {para} {f_n}"
#                 print(command)
#                 os.system(command)
