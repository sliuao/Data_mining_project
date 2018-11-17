import pandas as pd
import numpy as np

def get_time_dict():
    rng = pd.date_range('2014-05-26', '2014-06-24')
    print(rng)
    print('number of dates:', len(rng))
    time_dict = pd.Series(np.arange(len(rng)), index=rng)
    #print(time_dict['2013/10/30'])
    return time_dict

def get_duration(x,y):
    duration = pd.date_range(x,y)
    return len(duration)




class Enrollment():
    def __init__(self, filename):
        fin = open(filename)
        fin.next()

        self.enrollment_ids = []
        self.enrollment_info = {}
        self.user_info = {}
        self.user_enrollment_id = {}
        self.course_info = {}
        self.course_enrollment = {}

        for line in fin:
            enrollment_id, username, course_id = line.strip().split(',')
            if enrollment_id == 'enrollment_id':        # ignore the first row
                continue
            #print(enrollment_id)
            self.enrollment_ids.append(enrollment_id)
            self.enrollment_info[enrollment_id] = [username, course_id]

            if username not in self.user_info:
                self.user_info[username] = [course_id]
                self.user_enrollment_id[username] = [enrollment_id]
            else:
                self.user_info[username].append(course_id)
                self.user_enrollment_id[username].append(enrollment_id)

            if course_id not in self.course_info:
                self.course_info[course_id] = [username]
                self.course_enrollment[course_id] = [enrollment_id]
            else:
                self.course_info[course_id].append(username)
                self.course_enrollment[course_id].append(enrollment_id)
        print("load enrollment info over!")
        print("number of courses:", len(self.course_info))
        print("number of enrollments:", len(self.enrollment_info))
        print("number of students:", len(self.user_info))
        print("information of enrollment_id=1:", self.enrollment_info.get("1"))

class Log():

    '''
    enrollment_id
    time
    source
    event
    object
    '''

    def __init__(self, filename):
        fin = pd.read_csv(filename,header=0)
        #fin.next()
        #print(fin.header(5))

        self.enrollment_ids = []
        self.object_ids = []
        self.info = {}
        self.object_info = {}
        #self.event = {}

        self.file = fin


        # for i in range(len(fin)):

        #     #get value from different column in a line
        #     enrollment_id = fin.loc[i,'enrollment_id']
        #     time = fin.loc[i,'time']
        #     source = fin.loc[i,'source']
        #     event = fin.loc[i,'event']
        #     object_id = fin.loc[i,'object']


        #     if enrollment_id == 'enrollment_id':        # ignore the first row
        #         continue
        #     #print(enrollment_id)


        #     #a list store all enrollment id    
        #     if enrollment_id not in self.enrollment_ids:
        #         self.enrollment_ids.append(enrollment_id)
        #         #print(enrollment_id)

        #     #a list store all object id
        #     if object_id not in self.object_ids:
        #         self.object_ids.append(object_id)
        #     #print("222")

        #     if enrollment_id not in self.info:
        #         self.info[enrollment_id] = []
        #         self.info[enrollment_id].append([time, source, event, object_id])
        #     else:
        #         self.info[enrollment_id].append([time, source, event, object_id])
        #     #print("333")

        #     if object_id not in self.object_info:
        #         self.object_info[object_id] = []
        #         self.object_info[object_id].append([enrollment_id ,time, source, event])
        #     else:
        #         self.object_info[object_id].append([enrollment_id ,time, source, event])

        print("load log over!!!")
        print("number of rows: ", len(self.file))
        #print("number of different object id: ", len(self.object_ids))
        #print("information of enrollment_id=1:", self.info.get("1"))

class Truth():
    def __init__(self, filename):
        fin = open(filename)
        #fin.next()

        self.enrollment_ids = []
        self.truth_info = {}
        self.truth_label = []

        for line in fin:
            enrollment_id, truth = line.strip().split(',')
            #print(enrollment_id)

            self.enrollment_ids.append(enrollment_id)
            self.truth_info[enrollment_id] = [truth]
            self.truth_label.append(truth)

        print("number of different enrollment id: ", len(self.enrollment_ids))
        #print(self.truth_info["1"])
        #print(type(self.truth_info["4"]))


class Date():
    def __init__(self, filename):
        fin = open(filename)
        fin.next()

        self.course_ids = []
        self.course_duration = {}
        self.start = {}
        self.end = {}


        for line in fin:
            course_id, start, end = line.strip().split(',')

            if course_id == 'course_id':        # ignore the first row
                continue

            self.course_ids.append(course_id)
            duration = get_duration(start, end)
            self.course_duration[course_id] = [duration]
            self.start[course_id] = start
            self.end[course_id] = end

        print("load date over!!!")
        print("number of course id : ", len(self.course_ids))
        print("example data:")
        print(self.course_duration['DABrJ6O4AotFwuAbfo1fuMj40VmMpPGX'])

class Object():
    def __init__(self, filename):
        fin = open(filename)
        fin.next()

        self.course_ids = []
        #self.module_child = {}
        self.module_ids = []
        self.module_info = {}
        self.course_module = {}

        for line in fin:
            course_id, module_id, category, children, start = line.strip().split(',')
            
            if course_id == 'course_id':        # ignore the first row
                continue

            if course_id not in self.course_ids:
                self.course_ids.append(course_id)

            self.module_ids.append(module_id)

            if course_id not in self.course_module:
                self.course_module[course_id] = [module_id]
            else:
                self.course_module[course_id].append(module_id)

            self.module_info[module_id] = [category, children, start]


        print("load object over!!!")
        print("the number of different course id is : ",len(self.course_ids))
        print("the number of different module id is : ", len(self.module_ids))







#time_dict = get_time_dict()
#enrollment = Enrollment('data/train/enrollment_train.csv')
#log = Log('data/train/log_train.csv')
#truth = Truth('data/train/truth_train.csv')
#date = Date('data/date.csv')
#object = Object('data/object.csv')
