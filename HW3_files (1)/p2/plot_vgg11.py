import matplotlib.pyplot as plt
import csv
import numpy as np

times_rsp=[]
powers_rsp=[]
temps_rsp=[]

times_mc1=[]
powers_mc1=[]
temps_mc1=[]

runpow_rsp=[]
runpow_mc1=[]

df_rsp = open("log_raspberry_vgg11.csv")
df_mc1 = open("log_mc1_vgg11.csv")

df_rsp.readline()
df_mc1.readline()

for line in df_rsp:
    (time_stamp, total_power, temps_avg, temp) = line.split(',')
    xtime = float(time_stamp) - 1646070323.8911998
    times_rsp.append(float(xtime))
    powers_rsp.append(float(total_power))
    temps_rsp.append(float(temps_avg))
    if float(time_stamp) >= 1646070329.88807 and float(time_stamp) <= 1646071278.378308:
        runpow_rsp.append(float(total_power))

for line in df_mc1:
    (time_stamp, total_power, temps_avg, temp) = line.split(',')
    xtime = float(time_stamp) - 1646078030.299215
    times_mc1.append(float(xtime))
    powers_mc1.append(float(total_power))
    temps_mc1.append(float(temps_avg))
    if float(time_stamp) >= 1646078032.8226151 and float(time_stamp) <= 1646078692.6654227:
        runpow_mc1.append(float(total_power))

sum_runpow_rsp = sum(runpow_rsp)
sum_runpow_mc1 = sum(runpow_mc1)

print(len(runpow_rsp), len(runpow_mc1))

print("RSP energy (J) :", sum_runpow_rsp * (0.2))
print("MC1 energy (J) :", sum_runpow_mc1 * (0.2))

power_plot = plt.figure(1, figsize=(20,5))
plt.plot(times_mc1, powers_mc1, label='MC1 [W]')
plt.plot(times_rsp, powers_rsp, label='RSP [W]')
plt.title("System Power Consumption")
plt.xlabel("time[s]")
plt.ylabel("Power[W]")
plt.legend()
plt.grid()
plt.savefig("power_plot_vgg11.png")


power_plot = plt.figure(2, figsize=(20,5))
plt.plot(times_mc1, temps_mc1, label='MC1 [°C]')
plt.plot(times_rsp, temps_rsp, label='RSP [°C]')
plt.title("System Power Consumption")
plt.xlabel("time[s]")
plt.ylabel("Temperature[°C]")
plt.legend()
plt.grid()
plt.savefig("temp_plot_vgg11.png")


# rsp vgg 11
# start:  1646070329.88807
# end:  1646071278.378308
# Test Accuracy :  74.48
# Total time for inference : 915.9830 seconds
# Sec/Acc :  12.298374884750865

# mc1 vgg 11
# start: 