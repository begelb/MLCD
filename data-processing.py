import csv


'''
This file is used to convert Paul's data format to Brittany's data format.

'''
dim = 6
t = 'test'

h = open(f'data/ellipsoidal_{dim}d/{t}.csv', 'w')
# create the csv writer
writer = csv.writer(h)

data_list = []
with open(f'data/ellipsoidal_{dim}d/nf_{t}.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        row_list = []
        row = row[0]
        row = row.replace(' ', ',')
        for i in range(dim):
            row_list.append(row.split(',')[i])

       # print(row.split(','))
      #  x = row.split(',')[0]
    #    y = row.split(',')[1]
       # z = row.split(',')[2]
      #  w = row.split(',')[3]
      #  u = row.split(',')[4]
        l = row.split(',')[dim]
        #l = row.split(',')[6]
     #   row_list.append(x)  
     #   row_list.append(y)
       # row_list.append(z)
      #  row_list.append(w)
       # row_list.append(u)
        #row_list.append(n)
        if int(float(l)) == 0:
            row_list.append(0)
        elif int(float(l)) == 1:
            row_list.append(1)
        elif int(float(l)) == 2:
            row_list.append(2)
        elif int(float(l)) == 3:
            row_list.append(3)
        elif int(float(l)) == 4:
            row_list.append(4)
        elif int(float(l)) == 5:
            row_list.append(5)
        elif int(float(l)) == 6:
            row_list.append(6)
        

     #   w = row.split(',')[3]
     #   u = row.split(',')[4]
         #row_list.append(z)
     #   row_list.append(w)
     #   row_list.append(u)
        data_list.append(row_list)
        # write a row to the csv file
        writer.writerow(row_list)
    data = data_list


# close the file
h.close()
