import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import csv
import time

# change "xxx.xxxx" type of String to Float
def str_to_float(string):
    temp1 = 0.0
    temp2 = 0.0
    flag = False
    count = 0
    for char in string:
        if char != '.':
            if flag:
                temp2 *= 10
                temp2 += float(char)
                count += 1
            else:
                temp1 *= 10
                temp1 += float(char)
        else:
            flag = True

    for i in range(0, count):
        temp2 *= 0.1
            
    return round(temp1 + temp2, 7)

#===============================================================================

# change "Geometry" type String to "xxx.xxxxx" type of String
def geometry_to_coordinate(string): 
    temp1 = ''
    temp2 = ''
    flag = 0
    for char in string:
        if flag == 0 and char == '[':
            flag = 1
        elif flag == 1:
            if char >= '0' and char <= '9':
                temp1 = temp1 + char
            if char == '.':
                temp1 = temp1 + char
            if char == ',':
                flag = 2
        elif flag == 2:
            if char >= '0' and char <= '9':
                temp2 = temp2 + char
            if char == '.':
                temp2 = temp2 + char
            if char == ']':
                break
    
    return [str_to_float(temp1), str_to_float(temp2)]

def read_csv(file_name):
    return_list = []
    
    f = open(file_name, 'r', encoding = 'utf-8')

    i = 0
    for row in csv.reader(f):

        if row[9] == 'DELETED' or row[9] =='NOT_FOUND' or row[24] != 'DONE':
            continue

        if i:
            j = 0
            flag = False
            taskid = row[2]

            if row[8] == 'ADDED':
                temp_list = geometry_to_coordinate(row[10])
                x = temp_list[0]
                y = temp_list[1]

            else:
                x = str_to_float(row[16])
                y = str_to_float(row[17])
                
            area = str_to_float(row[18])
            teamid = int(row[15])
                
            for task in return_list:
                if task.get_taskid() == taskid:
                    task.add_ground(x, y, area, teamid)
                    flag = True
                    break
                j += 1
            if not flag:
                return_list.append(Task(taskid, x, y, area, teamid))        
        i += 1

    for task in return_list:
        task.set_etc()
        
    print('Task set finished\n')

    return return_list

#===============================================================================

# Task Class based on "TaskID"
class Task(): 

    # add one ground with initiation
    def __init__(self, new_id, x, y, area, teamid): 
        self.x_array = np.empty(shape = (0), dtype = 'float64')
        self.y_array = np.empty(shape = (0), dtype = 'float64')
        self.area_array = np.empty(shape = (0), dtype = 'float64')
        self.team_array = np.empty(shape = (0), dtype = int)

        self.taskid = new_id
        self.ground_count = 0
        self.add_ground(x, y, area, teamid)       

    # add Ground component
    def add_ground(self, x, y, area, teamid): 
        self.x_array = np.append(self.x_array, x)
        self.y_array = np.append(self.y_array, y)
        self.area_array = np.append(self.area_array, area)
        self.team_array = np.append(self.team_array, teamid)
        
        self.ground_count += 1

    # set how many team had taken this Task. and each total area of team had taken.
    def set_etc(self): 
        temp_list = []
        
        # count how many team is  here.
        for i in range(0, self.ground_count): 
            flag = False
            temp = self.team_array[i]
            for j in temp_list:
                if temp == j:
                    flag = True
            if not flag:
                temp_list.append(temp)

        self.team = len(temp_list)

        # set teamID -> (0  to k). ex) [727, 230, 91] -> [0, 1, 2]
        for i in range(0, self.ground_count): 
            for j in range(0, self.team):
                if self.team_array[i] == temp_list[j]:
                    self.team_array[i] = j

        #calcuate total_assigned_area
        self.team_assigned_area = np.zeros(shape = (self.team), dtype = 'float64')
        for i in range(0, self.ground_count):
            self.team_assigned_area[self.team_array[i]] += self.area_array[i]

    def get_team_number(self):
        return self.team

    def get_taskid(self):
        return self.taskid

    def get_ground_count(self):
        return self.ground_count

    def print(self):
        print('TaskID: ' + self.taskid)
        for i in range(0, self.ground_count):
            print('x: ', self.x_array[i], ' / y: ', self.y_array[i], ' / area: ', self.area_array[i])
        print('\nThis Task has ', self.ground_count, ' grounds.')
        print('End of data.\n\n')

#===============================================================================

class K_means():
    def __init__(self, Task, cycle = 10):
        self.cycle = cycle
        self.variation = -1.0

        self.k = Task.team
        self.data_size = Task.ground_count
        self.data_array = np.empty(shape = (self.data_size, 4), dtype = 'float64')
        self.need_area = Task.team_assigned_area.copy()

        self.data_array[:, 0] = Task.x_array.copy()
        self.data_array[:, 1] = Task.y_array.copy()
        self.data_array[:, 2] = Task.area_array.copy()
        self.data_array[:, 3] = Task.team_array.copy()

        self.old_team_array = np.empty(shape = (self.data_size), dtype = 'float64')
        #self.old_centers = np.empty(shape = (self.k, 2), dtype = 'float64')
        self.old_total_area = np.empty(shape = (0), dtype = 'float64')

    def set_random_points(self):
        self.centers = np.empty(shape = (self.k, 2), dtype = 'float64')

        i = 0
        temp_list = []
        # there must not be duplicated centers.
        while i < self.k:
            flag = False
            temp = random.randrange(0, self.data_size)
            for j in temp_list: #if duplicated.
                if j == temp:
                    flag = True
                    break
            if not flag:
                temp_list.append(temp) #else
                i += 1

        i = 0
        for number in temp_list: # make centers array
            self.centers[i][0] = self.data_array[number][0]
            self.centers[i][1] = self.data_array[number][1]
            i += 1

    def clustering(self):
        for i in range(0, self.data_size):
            temp1 = -1.0
            record = 0
            for j in range(0, self.k):
                x = self.data_array[i][0]
                y = self.data_array[i][1]

                tx = self.centers[j][0]
                ty = self.centers[j][1]
                
                temp2 = ((tx - x) * (tx - x)) + ((ty - y) * (ty - y))

                if temp1 == -1.0 or temp1 > temp2:
                    temp1 = temp2
                    record = j

            self.data_array[i][3] = record

    def center_adjusting(self):
        temp_array = np.zeros(shape = (self.k, 3), dtype = 'float64')

        #sum all of x and y value and count
        for i in range(0, self.data_size):
            temp = int(self.data_array[i][3])

            temp_array[temp][0] += self.data_array[i][0]
            temp_array[temp][1] += self.data_array[i][1]
            temp_array[temp][2] += 1

        # devide all of x and y to count
        for i in range(0, self.k):
            #if one of groups has 0 count of ground.
            #means, that group has disappeared.
            count = temp_array[i][2]
            if not count:
                return -1  #error code
            temp_array[i][0] /= count
            temp_array[i][1] /= count

        flag = False
        for i in range(0, self.k): # compare that center has moved
            if temp_array[i][0] != self.centers[i][0]:
                flag = True
                break
            if temp_array[i][1] != self.centers[i][1]:
                flag = True
                break

        if flag: #if it moved
            for i in range(0, self.k):
                self.centers[i][0] = temp_array[i][0]
                self.centers[i][1] = temp_array[i][1]
            return 1

        else: #else
            return 0

    def adjust_total_area(self):
        #sum all total_area of each groups.
        self.total_area = np.zeros(shape = (self.k), dtype = 'float64')   
        for i in range(0, self.data_size):
            self.total_area[int(self.data_array[i][3])] += self.data_array[i][2]

        # make 2di_array of distance between all dots and centers.
        distance_array = np.empty(shape = (self.k, self.data_size), dtype = 'float64')
        for i in range(0, self.k):
            tx = self.centers[i][0]
            ty = self.centers[i][1]
            for j in range(0, self.data_size):
                x = self.data_array[j][0]
                y = self.data_array[j][1]
                                
                distance_array[i][j] = ((tx - x) * (tx - x)) + ((ty - y) * (ty - y))

        # READ (README FILE/class K_means) to get more info.
        marker_array = np.zeros(shape = (self.data_size), dtype = int)

        flag = True
        while flag:
            temp_array = self.total_area.copy()
            for i in range(0, self.k):
                if self.total_area[i] < self.need_area[i]:
                    record1 = -1
                    record2 = -1
                    while self.total_area[i] < self.need_area[i]:
                        temp1 = -1.0
                        for j in range(0, self.data_size):
                            if int(self.data_array[j][3]) != i:
                                if not marker_array[j]:
                                    temp2 = distance_array[i][j]
                                    if temp1 == -1.0 or temp1 > temp2:
                                        temp1 = temp2
                                        record1 = j
                            else:
                                marker_array[j] = 1
                                
                        record2 = int(self.data_array[record1][3])
                        self.data_array[record1][3] = i
                        self.total_area[i] += self.data_array[record1][2]
                        self.total_area[record2] -= self.data_array[record1][2]

                        marker_array[record1] = 1
                        
                    self.data_array[record1][3] = record2
                    self.total_area[i] -= self.data_array[record1][2]
                    self.total_area[record2] += self.data_array[record1][2]

                    marker_array[record1] = 0

            for i in range(0, self.k):
                if temp_array[i] != self.total_area[i]:
                    flag = True
                    break
                
                else:
                    flag = False

    def calculate_variation(self, field = False):
        all_sum = 0.0
        for i in range(0, self.data_size):
            temp = int(self.data_array[i][3])
            
            x = self.data_array[i][0]
            y = self.data_array[i][1]
            tx = self.centers[temp][0]
            ty = self.centers[temp][1]

            all_sum += ((tx - x) * (tx - x)) + ((ty - y) * (ty - y))

        #if variation had changed, return 1
        if (self.variation == -1.0) or (self.variation > all_sum):
            self.variation = all_sum
            
            self.old_team_array = self.data_array[:, 3].copy()
            #self.old_centers = self.centers.copy()

            if field:
                self.old_total_area = self.total_area.copy()
            return 1
        
        #else, return 0
        return 0

    # calculate variation of "real devided  info."
    def get_default_variation(self):
        self.variation = -1.0
        self.set_random_points()
        self.center_adjusting()
        self.calculate_variation()
        return self.variation

    #normal k_means
    def random_kmeans(self):
        i = 0
        count = 0
        self.variation = -1.0
        while i < self.cycle:
            count += 1
            self.set_random_points()
            self.clustering()

            flag = 1
            while flag == 1: # while centers dont move anymore
                flag = self.center_adjusting()
                self.clustering()

            # if one of team didnt disappear.
            if flag != -1:
                # if variation change has occured.
                if self.calculate_variation():
                    i = 0
                #else
                else:
                    i += 1

        print('Variation:', self.variation, '\nthis time cycle number is:', count)

        return [count, self.variation]

    # special kmeans with "considering area need."
    def special_kmeans(self):
        i = 0
        count = 0
        self.variation = -1.0
        while i < self.cycle:
            count += 1
            self.set_random_points()
            self.clustering()

            flag = 1
            while flag == 1: # while centers dont move anymore
                flag = self.center_adjusting()
                self.clustering()

            # if one of team didnt disappear. and there are more than 1team.
            if flag != -1 and self.k > 1:
                self.adjust_total_area()
                self.center_adjusting()
                # if variation change has occured.
                if self.calculate_variation(True):
                    i = 0
                #else
                else:
                    i += 1

        print('Group needed area\n', self.need_area)
        print('Group total area\n', self.old_total_area)
        print('\nVariation:', self.variation, '\nthis time cycle number is:', count)

        return [count, self.variation]

    def get_x_array(self):
        return self.data_array[:, 0]
    
    def get_y_array(self):
        return self.data_array[:, 1]
    
    def get_old_team_array(self):
        return self.old_team_array
    
    
#===============================================================================

class Match_rate():
    def __init__(self, k, teams1):
        self.k = k
        self.teams1 = teams1.copy()
        self.data_size = teams1.size

    def add_team2(self, teams2):
        self.teams2 = teams2.copy()

    def calculate_match_rate(self):
        di1_temp_array = np.empty(shape = (self.k), dtype = int)
        
        di2_temp_array = np.empty(shape = (self.k, self.k), dtype = 'float64')

        for standard in range(0, self.k):
            for comparsion in range(0, self.k):
                temp = 0
                count = 0
                for i in range(0, self.data_size):
                    if self.teams1[i] == standard:
                        count += 1
                        if self.teams2[i] == comparsion:
                            temp += 1
                temp = temp / count * 100
                di2_temp_array[standard][comparsion] = temp

        for this in range(0, self.k):
            temp1 = -1.0
            record_i = -1.0
            record_j = -1.0
            for i in range(0, self.k):
                for j in range(0, self.k):
                    temp2 = di2_temp_array[i][j]
                    if temp2 != -1.0:
                        if temp1 < temp2:
                            temp1 = temp2
                            record_i = i
                            record_j = j
                            
            di1_temp_array[record_i] = record_j

            di2_temp_array[record_i, :] = -1.0
            di2_temp_array[:, record_j] = -1.0

        temp = 0.0
        for i in range(0, self.data_size):
            if di1_temp_array[int(self.teams1[i])] == int(self.teams2[i]):
                temp += 1

        temp = temp / self.data_size * 100
        print('Match rate is:', round(temp, 3), '%')

        return temp

#===============================================================================
            
if __name__ == "__main__":
    task_list = []

    task_list = read_csv('2.csv')

    f = open('result_sum.csv','a', newline='')
    wr = csv.writer(f)
    
    for task in task_list:
        kmeans = K_means(task, 100)

        dv = kmeans.get_default_variation()
        print('TaskID:', task.get_taskid())
        print('Task ground count:', task.get_ground_count())
        print('Task team count:', task.get_team_number())
        print('This is default variation', dv)
        df = pd.DataFrame({'x': kmeans.get_x_array(),
                            'y': kmeans.get_y_array(),
                            'color': kmeans.get_old_team_array()})
        sns.lmplot(x = 'x', y = 'y',
                   hue = 'color', data = df,
                   fit_reg = False,
                    scatter_kws = {"s": 10})
        rate = Match_rate(task.get_team_number(), kmeans.get_old_team_array())





        print('\nRANDOM K_MEANS')
        start = time.time()
        list1 = kmeans.random_kmeans()
        time1 = time.time() - start
        print('it took', round(time1, 3), 'secs.')
        df = pd.DataFrame({'x': kmeans.get_x_array(),
                           'y': kmeans.get_y_array(),
                            'color': kmeans.get_old_team_array()})
        sns.lmplot(x = 'x', y = 'y',
                   hue = 'color', data = df,
                   fit_reg = False,
                    scatter_kws = {"s": 10})
        rate.add_team2(kmeans.get_old_team_array())
        temp1 = rate.calculate_match_rate()
            




        print('\nSPECIAL K_MEANS')
        start = time.time()
        list2 = kmeans.special_kmeans()
        time2 = time.time() - start
        print('it took', round(time2, 3), 'secs.')
        df = pd.DataFrame({'x': kmeans.get_x_array(),
                           'y': kmeans.get_y_array(),
                            'color': kmeans.get_old_team_array()})
        sns.lmplot(x = 'x', y = 'y',
                   hue = 'color', data = df,
                   fit_reg = False,
                    scatter_kws = {"s": 10})
        rate.add_team2(kmeans.get_old_team_array())
        temp2 = rate.calculate_match_rate()
            
        plt.show(block = True)

        wr.writerow([task.get_taskid(), task.get_ground_count(),
                     task.get_team_number(), dv,
                     time1, list1[0], list1[1], temp1,
                     time2, list2[0], list2[1], temp2])

        print('========================================')

    f.close()
