import csv


'''
This file is used to convert Paul's data format to Brittany's data format.

'''
dim = 2
t = 'test'

#h = open(f'data/ellipsoidal_{dim}d/{t}.csv', 'w')
h = open(f'data/periodic_3_labels/data_periodic_shuffled_balanced.csv', 'w')
# create the csv writer
writer = csv.writer(h)

data_list = []

label_0_count = 0
label_1_count = 0
label_2_count = 0

with open(f'data/periodic_3_labels/data_periodic_shuffled.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        row_list = []
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
        
        if label_0_count < 5001 and label == 0:
            for i in range(dim):
                row_list.append(row[i])
            row_list.append(0)
            data_list.append(row_list)
            writer.writerow(row_list)


        if label_1_count < 5001 and label == 1:
            for i in range(dim):
                row_list.append(row[i])
            row_list.append(1)
            data_list.append(row_list)
            writer.writerow(row_list)

        if label_2_count < 5001 and label == 2:
            for i in range(dim):
                row_list.append(row[i])
            row_list.append(2)
            data_list.append(row_list)
            writer.writerow(row_list)

        print(label_1_count)
        print(label_0_count)
        print(label_2_count)

        if label_1_count > 5000 and label_0_count > 5000 and label_2_count > 5000:
            exit()


       # l = row.split(',')[dim]
       # if int(float(l)) == 0:
       #     row_list.append(0)
       # elif int(float(l)) == 1:
       #     row_list.append(1)
       # elif int(float(l)) == 2:
       #     row_list.append(2)
       # elif int(float(l)) == 3:
       #     row_list.append(3)
       # elif int(float(l)) == 4:
       #     row_list.append(4)
       # elif int(float(l)) == 5:
       #     row_list.append(5)
       # elif int(float(l)) == 6:
       #     row_list.append(6)
    data = data_list
    
h.close()
