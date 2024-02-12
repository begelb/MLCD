import csv

def save_formatted_data(labeled_pts_in_domain, num_of_pts, path):
    h = open(f'{path}/data.csv', 'w')
    writer = csv.writer(h)
    for i in range(num_of_pts):
        file_row = labeled_pts_in_domain[i, :]
        file_row = file_row.tolist()
        label = file_row[-1]
        file_row = file_row[:-1]
        file_row.append(int(label))
        writer.writerow(file_row)
    h.close()