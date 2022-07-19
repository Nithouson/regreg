from GridData5 import *
import xlwt

Side = 25
sid, tid = 0, 49

def recordnum(dataid):
    if dataid in [1, 5, 16]:
        return 46
    elif 20 <= dataid < 50:
        return 49
    return 45

noiselevel = ['low noise', 'medium noise', 'high noise']
method = ['KModels', 'AZP', 'RegKModels', 'GWR']
noiselabel = {'low noise':'l', 'medium noise':'m', 'high noise':'h'}

def Rand(zmap,coeff,eps=1e-6):
    Side = len(zmap)
    tp = fn = fp = tn = 0
    for i in range(Side*Side):
        for j in range(i+1,Side*Side):
            if abs(coeff[i]-coeff[j])<eps:
                if zmap[i//Side][i%Side] == zmap[j//Side][j%Side]:
                    tp+=1
                else:
                    fn+=1
            else:
                if zmap[i//Side][i%Side] == zmap[j//Side][j%Side]:
                    fp+=1
                else:
                    tn+=1
    npair = (Side*Side)*(Side*Side-1)//2
    assert npair == tp+fn+fp+tn
    return tp,fn,fp,tn,(tp+tn)/npair

outxls = xlwt.Workbook(encoding='utf-8')
sheets = {}
for noi in noiselevel:
    sheet = outxls.add_sheet(noi)
    sheets[noi] = sheet
    sheet.write(0, 0, label='Data')
    for m in range(len(method)):
        sheet.write(0, 5*m+1, label=method[m]+'_TP')
        sheet.write(0, 5*m+2, label=method[m]+'_FN')
        sheet.write(0, 5*m+3, label=method[m]+'_FP')
        sheet.write(0, 5*m+4, label=method[m]+'_TN')
        sheet.write(0, 5*m+5, label=method[m]+'_Rand')

for id in range(sid,tid+1):
    resfile = open("../log/result_" + str(recordnum(id)) + "_" + str(id) + ".txt")
    for noi in noiselevel:
        print(str(id) + noiselabel[noi])
        gt = open("../synthetic/edis_" + str(id) + noiselabel[noi] + ".txt")
        Xarr, Yarr, coeff = input_data_dis(gt, Side)
        sheets[noi].write(id + 1, 0, label='data_' + str(id))

        for m in range(4):
            zonemap = [[-1 for c in range(Side)] for r in range(Side)]
            for r in range(Side):
                zoneline = resfile.readline().split(" ")
                while len(zoneline) <= 1:
                    zoneline = resfile.readline().split(" ")
                zoneline.remove('\n')
                assert len(zoneline) == Side
                for c in range(Side):
                    zonemap[r][c] = int(zoneline[c])
            n_regions = max([max(zonemap[r]) for r in range(Side)]) + 1
            regions = [[] for z in range(n_regions)]
            for r in range(Side):
                for c in range(Side):
                    regions[zonemap[r][c]].append(Pos_Encode(r, c, Side))

            rcoeff = [[-9999, -9999] for z in range(n_regions)]
            for z in range(n_regions):
                zoneline = resfile.readline().split(" ")
                while len(zoneline) <= 1:
                    zoneline = resfile.readline().split(" ")
                zoneline.remove('\n')
                assert len(zoneline) == 2
                for v in range(2):
                    rcoeff[z][v] = float(zoneline[v])

            tp,fn,fp,tn,randi = Rand(zonemap,coeff[:,1])
            sheets[noi].write(id+1, 5 * m + 1, label=tp)
            sheets[noi].write(id+1, 5 * m + 2, label=fn)
            sheets[noi].write(id+1, 5 * m + 3, label=fp)
            sheets[noi].write(id+1, 5 * m + 4, label=tn)
            sheets[noi].write(id+1, 5 * m + 5, label=randi)

outxls.save("Rand.xls")

