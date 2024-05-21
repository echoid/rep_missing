import os

#data_list = ["banknote", "climate_model_crashes", "connectionist_bench_sonar", "qsar_biodegradation", "yeast"]
data_list = ['wine_quality_white','yacht_hydrodynamics','concrete_compression']
type_list = ["mnar", "mar", "mcar"]
para_list = [0.05,0.1,0.3,0.5]


for data in data_list:
    for types in type_list:
        for para in para_list:
            command = f"python genRBF/main_demo_reg.py {data} {types} {para}"
            print(command)
            os.system(command)
