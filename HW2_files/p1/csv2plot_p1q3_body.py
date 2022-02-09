import matplotlib.pyplot as plt
import csv

times=[]
powers=[]
temps=[]


df = open("log_infinite_p1_q3_bodytrack.csv")

for line in df:
    (time,power,usage_c0,usage_c1,usage_c2,usage_c3,core4,core5,core6,core7,temp4,temp5,temp6,temp7, dump) = line.split(',')
    xtime = float(time) - 1644445763.3950467
    times.append(float(xtime))
    powers.append(float(power))

    # core4s.append(float(core4)*100)
    # core5s.append(float(core5)*100)
    # core6s.append(float(core6)*100)
    # core7s.append(float(core7)*100)
    temps.append(max(float(temp4),float(temp5),float(temp6),float(temp7)))



power_plot = plt.figure(1, figsize=(20,5))
plt.plot(times, powers, label='Power[W]')
plt.title("System Power Consumption")
plt.xlabel("time[s]")
plt.ylabel("Power[W]")
plt.legend()
plt.grid()
plt.savefig("power_plot_p1q3_body.png")


core_plot = plt.figure(2, figsize=(20,5))
plt.plot(times, temps, 'g', label='Temperature')
plt.title("Temperature")
plt.xlabel("time[s]")
plt.ylabel("Max Temperature[°C]")
plt.legend()
plt.grid()
plt.savefig("temp_plot_p1q3_body.png")



