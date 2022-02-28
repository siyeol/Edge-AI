import matplotlib.pyplot as plt
import csv
import numpy

runpow_rsp=[]
runpow_mc1=[]

df_rsp = open("log_raspberry_mobilenetv1.csv")
df_mc1 = open("log_mc1_mobilenetv1.csv")

df_rsp.readline()
df_mc1.readline()

for line in df_rsp:
    (time_stamp, total_power, temps_avg, temp) = line.split(',')
    if float(time_stamp) >= 1646075546.3502688 and float(time_stamp) <= 1646076083.914866:
        runpow_rsp.append(float(total_power))

for line in df_mc1:
    (time_stamp, total_power, temps_avg, temp) = line.split(',')
    if float(time_stamp) >= 1646083121.9592168 and float(time_stamp) <= 1646083568.184102:
        runpow_mc1.append(float(total_power))

sum_runpow_rsp = sum(runpow_rsp)
sum_runpow_mc1 = sum(runpow_mc1)

print(len(runpow_rsp), len(runpow_mc1))

print("RSP energy (J) :", sum_runpow_rsp * (0.2))
print("MC1 energy (J) :", sum_runpow_mc1 * (0.2))

