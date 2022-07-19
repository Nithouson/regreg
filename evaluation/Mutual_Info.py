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
method = ['KM', 'AZP', 'RKM', 'GWR']
noiselabel = {'low noise':'l', 'medium noise':'m', 'high noise':'h'}


def Mutual_Info(reg_p,reg_r):
    n = sum([len(r) for r in reg_r])
    assert n == sum([len(r) for r in reg_p])
    hp = -sum([(len(r) / n) * math.log2(len(r) / n) for r in reg_p])
    hr = -sum([(len(r) / n) * math.log2(len(r) / n) for r in reg_r])
    reg_ps = [set(r) for r in reg_p]
    reg_rs = [set(r) for r in reg_r]
    mi = 0
    for rp in reg_ps:
        for rr in reg_rs:
            n_int = len(rp.intersection(rr))
            if n_int == 0:
                continue
            mi += (n_int/n) * math.log2(n*n_int/(len(rp)*len(rr)))
    nmi = 2*mi/(hp+hr)
    return hp,hr,mi,nmi


outxls = xlwt.Workbook(encoding='utf-8')
sheets = {}
for noi in noiselevel:
    sheet = outxls.add_sheet(noi)
    sheets[noi] = sheet
    sheet.write(0, 0, label='Data')
    for m in range(len(method)):
        sheet.write(0, 4*m+1, label=method[m]+'_hp')
        sheet.write(0, 4*m+2, label=method[m]+'_hr')
        sheet.write(0, 4*m+3, label=method[m]+'_MI')
        sheet.write(0, 4*m+4, label=method[m]+'_NMI')

for id in range(sid,tid+1):
    resfile = open("../log/result_" + str(recordnum(id)) + "_" + str(id) + ".txt")
    for noi in noiselevel:
        print(str(id) + noiselabel[noi])
        gt = open("../synthetic/edis_" + str(id) + noiselabel[noi] + ".txt")
        Xarr, Yarr, coeff = input_data_dis(gt, Side)
        regions = [[] for z in range(5)]
        for u in range(Side*Side):
            regions[int(round(coeff[u, 1]))].append(u)
        assert sum([len(reg) for reg in regions]) == Side*Side
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
            regions_pred = [[] for z in range(n_regions)]
            for r in range(Side):
                for c in range(Side):
                    regions_pred[zonemap[r][c]].append(Pos_Encode(r, c, Side))

            rcoeff = [[-9999, -9999] for z in range(n_regions)]
            for z in range(n_regions):
                zoneline = resfile.readline().split(" ")
                while len(zoneline) <= 1:
                    zoneline = resfile.readline().split(" ")
                zoneline.remove('\n')
                assert len(zoneline) == 2
                for v in range(2):
                    rcoeff[z][v] = float(zoneline[v])

            hp, hr, mi, nmi = Mutual_Info(regions_pred,regions)
            sheets[noi].write(id + 1, 4 * m + 1, label=hp)
            sheets[noi].write(id + 1, 4 * m + 2, label=hr)
            sheets[noi].write(id+1, 4 * m + 3, label=mi)
            sheets[noi].write(id+1, 4 * m + 4, label=nmi)

outxls.save("MutualInfo.xls")