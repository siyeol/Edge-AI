import matplotlib.pyplot as plt
import csv
import numpy

times_rsp=[]
powers_rsp=[]
temps_rsp=[]

times_mc1=[]
powers_mc1=[]
temps_mc1=[]


df_rsp = open("log_raspberry_vgg16.csv")
df_mc1 = open("log_mc1_vgg16.csv")

df_rsp.readline()
df_mc1.readline()

for line in df_rsp:
    (time_stamp, total_power, temps_avg, temp) = line.split(',')
    xtime = float(time_stamp) - 1646072673.128221
    times_rsp.append(float(xtime))
    powers_rsp.append(float(total_power))
    temps_rsp.append(float(temps_avg))

for line in df_mc1:
    (time_stamp, total_power, temps_avg, temp) = line.split(',')
    xtime = float(time_stamp) - 1646080363.312095
    times_mc1.append(float(xtime))
    powers_mc1.append(float(total_power))
    temps_mc1.append(float(temps_avg))


power_plot = plt.figure(1, figsize=(20,5))
plt.plot(times_mc1, powers_mc1, label='MC1 [W]')
plt.plot(times_rsp, powers_rsp, label='RSP [W]')
plt.title("System Power Consumption")
plt.xlabel("time[s]")
plt.ylabel("Power[W]")
plt.legend()
plt.grid()
plt.savefig("power_plot_vgg16.png")


power_plot = plt.figure(2, figsize=(20,5))
plt.plot(times_mc1, temps_mc1, label='MC1 [°C]')
plt.plot(times_rsp, temps_rsp, label='RSP [°C]')
plt.title("System Power Consumption")
plt.xlabel("time[s]")
plt.ylabel("Temperature[°C]")
plt.legend()
plt.grid()
plt.savefig("temp_plot_vgg16.png")
