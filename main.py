import os
import pandas as pd
import math
import sys
import numpy as np
import transmat as transmat
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import mixture
from sklearn import preprocessing
# from mlxtend.feature_selection import SequentialFeatureSelector

# turn off warnings
pd.options.mode.chained_assignment = None

number_of_users = 29
usuario = 1000
total_signatures = 56
genuines = 46


def carica_firma(numero_utente, numero_firma, label):

    path = 'xLongSignDB/usuario'+str(usuario+numero_utente)+'/'

    if numero_firma < 10:
        path = os.path.abspath(path + str(usuario + numero_utente) + label + '_0' + str(numero_firma) + '.svc')

    else:
        path = os.path.abspath(path + str(usuario + numero_utente) + label + '_' + str(numero_firma) + '.svc')

    df = pd.read_csv(path, header=None, skiprows=1, sep=' ')

    df.columns = ['X', 'Y', 'TIMESTAMP', 'PENISUP', 'AZIMUTH', 'ALTITUDE', 'Z']

    return df


# derivata_colonna:
# prende in input la colonna della tabella e ne restituisce la derivata utilizzando una regressione di secondo grado
def derivata_colonna(colonna):

    colonna_list = colonna.tolist()
    colonna_list.append(colonna[len(colonna)-1])
    colonna_list.append(colonna[len(colonna)-1])
    colonna_list.insert(0, colonna_list[0])
    colonna_list.insert(0, colonna_list[0])
    derivata = [None] * (len(colonna))

    for n in range(2, len(colonna)+2):
        derivata[n-2] = ((colonna_list[n+1]-colonna_list[n-1]+2*(colonna_list[n+2]-colonna_list[n-2]))/10)

    return derivata

def angolo_punti_consecutivi(X_list,Y_list):

    lunghezza = len(X_list)
    Angle_list = [None] * (lunghezza)
    X_list.append(X_list[lunghezza-1])
    Y_list.append(Y_list[lunghezza-1])

    for n in range(0,lunghezza):

        if X_list[n+1]-X_list[n] == 0:
            Angle_list[n] = np.arctan(np.Infinity)
        else:
            Angle_list[n]=np.arctan((Y_list[n+1]-Y_list[n])/(X_list[n+1]-X_list[n]))


    return Angle_list

def ratio_5window(up_list,x_list):

    up_list.append(0)
    up_list.append(0)
    up_list.insert(0,0)
    up_list.insert(0,0)

    lunghezza = len(x_list)
    x_list.append(x_list[lunghezza - 1])
    x_list.append(x_list[lunghezza - 1])
    x_list.insert(0,x_list[0])
    x_list.insert(0,x_list[0])

    ratio_list = [None]*(lunghezza)

    for n in range(0,lunghezza):

        stroke_len = np.sum(up_list[n:n+4])
        width = np.max(x_list[n:n+4]) - np.min(x_list[n:n+4])

        if stroke_len == 0:
            ratio_list[n] = 0
        else:
            if width == 0:
                ratio_list[n] = stroke_len
            else:
                ratio_list[n] = stroke_len/width

    return ratio_list


def ratio_7window(up_list, x_list):

    up_list.append(0)
    up_list.append(0)
    up_list.append(0)
    up_list.insert(0, 0)
    up_list.insert(0, 0)
    up_list.insert(0, 0)

    lunghezza = len(x_list)

    x_list.append(x_list[lunghezza - 1])
    x_list.append(x_list[lunghezza - 1])
    x_list.append(x_list[lunghezza - 1])
    x_list.insert(0, x_list[0])
    x_list.insert(0, x_list[0])
    x_list.insert(0, x_list[0])


    ratio_list = [None] * (lunghezza)

    for n in range(0, lunghezza):
        stroke_len = np.sum(up_list[n:n + 6])
        width = np.max(x_list[n:n + 6]) - np.min(x_list[n:n + 6])


        if stroke_len == 0:
            ratio_list[n] = 0
        else:
            if width == 0:
                ratio_list[n] = stroke_len
            else:
                ratio_list[n] = stroke_len/width

    return ratio_list

table = carica_firma(1, 1, 'sg')

del table['TIMESTAMP']
Up_List = table['PENISUP']
del table['PENISUP']
del table['AZIMUTH']
del table['ALTITUDE']

dx_list = derivata_colonna(table['X'])

dy_list = derivata_colonna(table['Y'])
tan_ang = [None] * (len(table))
velocity = [None] * (len(table))
log_k = [None] * (len(table))
c_list = [None] * (len(table))
acceleration = [None] * (len(table))

for n in range(0, len(table)):
    if dx_list[n] == 0:
        tan_ang[n] = 0
    else:
        tan_ang[n] = math.atan(dy_list[n]/dx_list[n])

table['TanAng'] = tan_ang
dt_list = derivata_colonna(table['TanAng'])

for n in range(0, len(table)):
    velocity[n] = math.sqrt(dy_list[n]**2+dx_list[n]**2)

table['v'] = velocity
dv_list = derivata_colonna(table['v'])

for n in range(0, len(table)):
    if dt_list[n] <= 0.0:
        log_k[n] = math.log(sys.maxsize)
    else:
        if velocity[n] <= 0.0:
            log_k[n] = math.log(0.01)
        else:
            log_k[n] = math.log(velocity[n]/dt_list[n])

table['logK'] = log_k
dlogk_list = derivata_colonna(table['logK'])

for n in range(0, len(table)):
    c_list[n] = dt_list[n]*velocity[n]
    acceleration[n] = math.sqrt(c_list[n]**2+dv_list[n]**2)

table['acc'] = acceleration
da_list = derivata_colonna(table['acc'])

table['dX'] = dx_list
table['dY'] = dy_list
table['dZ'] = derivata_colonna(table['Z'])
table['dTan'] = dt_list
table['dv'] = dv_list
table['dLog'] = dlogk_list
table['dAcc'] = da_list
table['ddX'] = derivata_colonna(table['dX'])
table['ddY'] = derivata_colonna(table['dY'])

vel_list = table['v'].tolist()
vel_list.append(vel_list[len(vel_list) - 1])
vel_list.append(vel_list[len(vel_list) - 1])
vel_list.insert(0, vel_list[0])
vel_list.insert(0, vel_list[0])
vel_ratio = [None] * (len(table))

for n in range(0, len(table)):
    vmin = np.amin(vel_list[n:n+4])
    vmax = np.amax(vel_list[n:n+4])
    if vmax == 0:
        vel_ratio[n] = 0
    else:
        vel_ratio[n] = vmin/vmax

table['vRatio'] = vel_ratio

print(table)

"""d = preprocessing.normalize(table)
table = pd.DataFrame(d, columns=table.columns)"""

"""x = table.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
table = pd.DataFrame(x_scaled, columns=table.columns)"""
"""
scaler = preprocessing.StandardScaler()
transform = scaler.fit_transform(table.values)
table = pd.DataFrame(transform, columns=table.columns)

mean = "{:.2f}".format(table.values.mean())
print("Mean :\n", mean)
sd = np.std(table.values)
print("Standard Deviation :\n", sd)
print(table)
"""


table['ANG']= angolo_punti_consecutivi(table['X'].tolist(),table['Y'].tolist())
table['D_ANG']= derivata_colonna(table['ANG'])
table['SIN'] = np.sin(table['ANG'])
table['COS'] = np.cos(table['ANG'])
table['5_WIN'] = ratio_5window(Up_List.tolist(),table['X'].tolist())
table['7_WIN'] = ratio_7window(Up_List.tolist(),table['X'].tolist())


print(table)
model = mixture.GaussianMixture(n_components=32,random_state=0)
print(SequentialFeatureSelector(model,n_features_to_select=9,direction='forward'))

"""

for i in range(1,number_of_users+1):

    usuario +=1
    print(usuario)
    cc = 1

    for k in range(1, total_signatures+1):

        if k <= genuines:

            if k < 10:
                path = 'C:/Users/feder/Desktop/xLongSignDB/usuario' + str(usuario) + '/' + str(usuario) + 'sg_0'+str(k)+'.svc'
                table = pd.read_csv(path)

            else:
                path = 'C:/Users/feder/Desktop/xLongSignDB/usuario' + str(usuario) + '/' + str(usuario) +'sg_'+str(k)+'.svc'
                table = pd.read_csv(path)
        else:

            if cc < 10:
                path = 'C:/Users/feder/Desktop/xLongSignDB/usuario' + str(usuario) + '/' + str(usuario) +'ss_0' + str(cc) + '.svc'
                table = pd.read_csv(path)
                cc += 1
            else:
                path = 'C:/Users/feder/Desktop/xLongSignDB/usuario' + str(usuario) + '/' + str(usuario) + 'ss_' + str(cc) + '.svc'
                table = pd.read_csv(path)

        print(path)
        print(table)
        null_columns = table.isnull().values.any()
        if(null_columns == True):
            print(null_columns)

"""
