import matplotlib.pyplot as plt
import csv

times=[]
powers=[]
core4s=[]
core5s=[]
core6s=[]
core7s=[]
temp4s=[]
temp5s=[]
temp6s=[]
temp7s=[]

df = open("log_infinite.csv")

for line in df:
    (time,power,usage_c0,usage_c1,usage_c2,usage_c3,core4,core5,core6,core7,temp4,temp5,temp6,temp7, dump) = line.split(',')
    xtime = float(time) - 1644439973.2520738
    times.append(float(xtime))
    powers.append(float(power))

    core4s.append(float(core4)*100)
    core5s.append(float(core5)*100)
    core6s.append(float(core6)*100)
    core7s.append(float(core7)*100)

    temp4s.append(float(temp4))
    temp5s.append(float(temp5))
    temp6s.append(float(temp6))
    temp7s.append(float(temp7))


power_plot = plt.figure(1, figsize=(20,5))
plt.plot(times, powers, label='Power[W]')
plt.title("System Power Consumption")
plt.xlabel("time[s]")
plt.ylabel("Power[W]")
plt.legend()
plt.grid()
plt.savefig("power_plot.png")


core_plot = plt.figure(2, figsize=(20,5))
plt.plot(times, core4s, 'g', label='Core 4')
plt.plot(times, core5s, 'b', label='Core 5')
plt.plot(times, core6s, 'r', label='Core 6')
plt.plot(times, core7s, 'y', label='Core 7')
plt.title("Core Usage")
plt.xlabel("time[s]")
plt.ylabel("Core Usage[%]")
plt.legend()
plt.grid()
plt.savefig("core_plot.png")


core_plot = plt.figure(3, figsize=(20,5))
plt.plot(times, temp4s, 'g', label='Temp 4')
plt.plot(times, temp5s, 'b', label='Temp 5')
plt.plot(times, temp6s, 'r', label='Temp 6')
plt.plot(times, temp7s, 'y', label='Temp 7')
plt.title("Temperature")
plt.xlabel("time[s]")
plt.ylabel("Temperature[°C]")
plt.legend()
plt.grid()
plt.savefig("temp_plot.png")



