import csv


'''
This file is used to convert Paul's data format to Brittany's data format.

'''
dim = 4
t = 'test'

#h = open(f'data/ellipsoidal_{dim}d/{t}.csv', 'w')
h = open(f'data/ellipsoidal_4d/test.csv', 'w')
# create the csv writer
writer = csv.writer(h)

data_list = []

label_0_count = 0
label_1_count = 0
label_2_count = 0
label_3_count = 0

with open(f'data/ellipsoidal_4d/nf_test.csv', newline='') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        row_list = []
        print('row: ', row)
       # print('row: ', row)
       # row = row[0]
      #  row = row.replace(' ', ',')
       # print(row.split(','))
        #l = row.split(',')[dim]
        l = row[dim]
        label = int(float(l))
        
        if label == 0:
            label_0_count += 1
        if label == 1:
            label_1_count += 1
        if label == 2:
            label_2_count += 1
        if label == 3:
            label_3_count += 1
        
        if label_0_count <= 10000 and label == 0:
            for i in range(dim):
                row_list.append(row[i])
            row_list.append(0)
            data_list.append(row_list)
            writer.writerow(row_list)

        if label_1_count <= 10000 and label == 1:
            for i in range(dim):
                row_list.append(row[i])
            row_list.append(1)
            data_list.append(row_list)
            writer.writerow(row_list)

        if label_2_count <= 10000 and label == 2:
            for i in range(dim):
                row_list.append(row[i])
            row_list.append(2)
            data_list.append(row_list)
            writer.writerow(row_list)

        if label_3_count <= 10000 and label == 3:
            for i in range(dim):
                row_list.append(row[i])
            row_list.append(3)
            data_list.append(row_list)
            writer.writerow(row_list)

    print('1', label_1_count)
    print('0', label_0_count)
    print('2', label_2_count)
    print('3', label_3_count)
    print('----')

    data = data_list
    
h.close()
