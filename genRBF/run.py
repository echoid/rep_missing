import os

data_list = ["banknote", "climate_model_crashes", "connectionist_bench_sonar", "qsar_biodegradation", "yeast"]
data_list = ['concrete_compression','wine_quality_red','wine_quality_white','yacht_hydrodynamics']
type_list = ["diffuse", "logistic", "mar", "mcar"]

for data in data_list:
    for types in type_list:
        #command = f"python main_demo.py ./{data}_{types}/"
        command = f"python main_reg.py ./{data}_{types}/"
        os.system(command)