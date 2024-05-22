import os

data_list = ["banknote", "yeast", "climate_model_crashes"]
#data_list = ['wine_quality_white','yacht_hydrodynamics','concrete_compression']
type_list = ["mnar", "mar", "mcar"]
para_list = [0.05,0.1,0.3,0.5]

data_list = ['wine_quality_white','yacht_hydrodynamics','concrete_compression']
type_list = ["mnar", "mar", "mcar"]
para_list = [0.05,0.1,0.3,0.5]



for data in data_list:
    for types in type_list:
        for para in para_list:
            #command = f"python genRBF/main_demo_reg.py {data} {types} {para}"
            print(data,types,para)
            #command = f"python kernal_pca_run.py {data} {types} {para}"
            #command = f"python baseline.py {data} {types} {para}"
            #command = f"python ppca_run.py {data} {types} {para}"
            #command = f"python ik_run.py {data} {types} {para}"
            command = f"python kernal_pca_run_reg.py {data} {types} {para}"
            #command = f"python baseline_reg.py {data} {types} {para}"
            os.system(command)



# command = f"git add ."
# print(command)
# os.system(command)

# command = f'git commit -m "update IK"'
# print(command)
# os.system(command)

# command = f'git push'
# print(command)
# os.system(command)