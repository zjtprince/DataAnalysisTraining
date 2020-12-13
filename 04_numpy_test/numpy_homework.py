#!/usr/bin/python
#vim: set fileencoding:utf-8



import numpy as np
import numpy.lib.recfunctions as rfn

type = np.dtype({"names":['name','chinese','english','math'],
                 "formats":["U32","i",'i','i']})

score = np.array([('张飞',66,65,30),
                  ('关羽',95,85,98),
                  ('赵云',93,92,96),
                  ('关羽',95,85,98),
                  ('黄忠',90,88,77),
                  ('典韦',80,90,90)

                  ],dtype=type)
print(score)

#语文、英语、数学中的平均成绩、最小成绩、最大成绩、方差、标准差
#语文均成绩、最小成绩、最大成绩、方差、标准差

chinese_score = score[:]['chinese']
print("语文平均值\t", np.average(chinese_score))
print("语文最小值\t", np.amin(chinese_score))
print("语文最大值\t", np.amax(chinese_score))
print("语文方差  \t", np.var(chinese_score))
print("语文标准差\t", np.std(chinese_score))
##数学英语类似


for col in score.dtype.names:
# print(col)
    if col is "name":
        continue
    print("mean of {}: {}".format(col, score[col].mean()))
    print("min  of {}: {}".format(col, score[col].min()))
    print("max  of {}: {}".format(col, score[col].max()))
    print("var  of {}: {}".format(col, score[col].var()))
    print("std  of {}: {}".format(col, score[col].std()))

total = score[:]['chinese'] + score[:]['english'] + score[:]['math']
score_total = np.lib.recfunctions.append_fields(score, names="total",data=total)
print(np.sort(score_total,order='total'))

rank = sorted(score,key=lambda x : x[1]+x[2]+x[3], reverse=True)
print(rank)


