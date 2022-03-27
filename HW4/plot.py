import matplotlib.pyplot as plt

vgg11_acc= [0.42480000853538513, 0.5285000205039978, 0.5910999774932861, 0.6050000190734863, 0.6477000117301941, 0.6976000070571899, 0.704200029373169, 0.7088000178337097, 0.732699990272522, 0.7324000000953674, 0.7336000204086304, 0.7440999746322632, 0.7289999723434448, 0.7470999956130981, 0.7504000067710876, 0.7476000189781189, 0.7488999962806702, 0.7501000165939331, 0.7592999935150146, 0.739799976348877, 0.757099986076355, 0.7591000199317932, 0.7559999823570251, 0.7512000203132629, 0.7542999982833862, 0.7491000294685364, 0.7662000060081482, 0.7394999861717224, 0.7555000185966492, 0.7605000138282776, 0.7612000107765198, 0.7585999965667725, 0.7533000111579895, 0.7548999786376953, 0.7626000046730042, 0.7633000016212463, 0.7563999891281128, 0.7437999844551086, 0.7508999705314636, 0.7563999891281128, 0.7565000057220459, 0.7653999924659729, 0.7541000247001648, 0.7635999917984009, 0.7544999718666077, 0.7682999968528748, 0.7620999813079834, 0.7476999759674072, 0.7651000022888184, 0.7684000134468079, 0.7580999732017517, 0.7702000141143799, 0.7594000101089478, 0.7663999795913696, 0.7642999887466431, 0.767300009727478, 0.7529000043869019, 0.7648000121116638, 0.7642999887466431, 0.7541999816894531, 0.7671999931335449, 0.7652999758720398, 0.7594000101089478, 0.760699987411499, 0.7669000029563904, 0.7620999813079834, 0.7594000101089478, 0.7736999988555908, 0.7634000182151794, 0.7630000114440918, 0.7547000050544739, 0.7689999938011169, 0.7750999927520752, 0.7616999745368958, 0.7671999931335449, 0.7723000049591064, 0.7651000022888184, 0.76419997215271, 0.771399974822998, 0.7663999795913696, 0.7644000053405762, 0.7698000073432922, 0.7687000036239624, 0.7620000243186951, 0.7688000202178955, 0.7675999999046326, 0.7651000022888184, 0.774399995803833, 0.7635999917984009, 0.7695000171661377, 0.774399995803833, 0.7667999863624573, 0.7666000127792358, 0.7795000076293945, 0.775600016117096, 0.7720000147819519, 0.7702999711036682, 0.7630000114440918, 0.7684000134468079, 0.7774999737739563]
vgg16_acc= [0.39399999380111694, 0.5006999969482422, 0.5771999955177307, 0.5848000049591064, 0.6743000149726868, 0.7001000046730042, 0.7218999862670898, 0.7049000263214111, 0.7333999872207642, 0.7304999828338623, 0.7401999831199646, 0.736299991607666, 0.7455999851226807, 0.7465000152587891, 0.7411999702453613, 0.7394999861717224, 0.7421000003814697, 0.7415000200271606, 0.7477999925613403, 0.7354000210762024, 0.7556999921798706, 0.7562000155448914, 0.7433000206947327, 0.7537000179290771, 0.7530999779701233, 0.7558000087738037, 0.7452999949455261, 0.7581999897956848, 0.7559999823570251, 0.7462999820709229, 0.7598999738693237, 0.7580999732017517, 0.7537000179290771, 0.7544000148773193, 0.7612000107765198, 0.7631000280380249, 0.7569000124931335, 0.7549999952316284, 0.7613000273704529, 0.7616999745368958, 0.7609000205993652, 0.7667999863624573, 0.7591999769210815, 0.766700029373169, 0.7574999928474426, 0.7556999921798706, 0.7669000029563904, 0.7681000232696533, 0.767300009727478, 0.7670999765396118, 0.7445999979972839, 0.7580999732017517, 0.7677000164985657, 0.766700029373169, 0.7538999915122986, 0.7657999992370605, 0.7714999914169312, 0.7738000154495239, 0.7639999985694885, 0.776199996471405, 0.7653999924659729, 0.7631999850273132, 0.7742999792098999, 0.761900007724762, 0.772599995136261, 0.7748000025749207, 0.777999997138977, 0.7689999938011169, 0.7685999870300293, 0.7610999941825867, 0.7688000202178955, 0.7753999829292297, 0.7645999789237976, 0.7746999859809875, 0.7742000222206116, 0.7712000012397766, 0.7620999813079834, 0.7742000222206116, 0.7768999934196472, 0.7698000073432922, 0.7738000154495239, 0.7806000113487244, 0.7677000164985657, 0.777999997138977, 0.7731000185012817, 0.7827000021934509, 0.7824000120162964, 0.7734000086784363, 0.776199996471405, 0.7786999940872192, 0.7736999988555908, 0.7813000082969666, 0.7799000144004822, 0.7764000296592712, 0.7824000120162964, 0.7735000252723694, 0.7860999703407288, 0.7785999774932861, 0.777899980545044, 0.781499981880188]

epochs = list(range(1,101))
plt.plot(epochs, vgg11_acc, label='VGG11')
plt.plot(epochs, vgg16_acc, label='VGG16')
plt.title("Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Testing Accuracy[%]")
plt.legend()
plt.grid()
plt.savefig("test_acc_plot.png")
