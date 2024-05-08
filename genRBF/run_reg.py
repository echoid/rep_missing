import os

#data_list = ["banknote", "climate_model_crashes", "connectionist_bench_sonar", "qsar_biodegradation", "yeast"]
data_list = ['concrete_compression','wine_quality_red','wine_quality_white','yacht_hydrodynamics']
type_list = ["diffuse", "logistic", "mar", "mcar"]
para_list = [0.3,0.5,0.7]
full_norm = ["full","norm"]

for data in data_list:
    for types in type_list:
        for para in para_list:
            for f_n in full_norm:
            #command = f"python main_demo.py ./{data}_{types}/"
                command = f"python genRBF/main_demo2.py {data} {types} {para} {f_n}"
                print(command)
                os.system(command)
