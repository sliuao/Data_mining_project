import read_data
import pandas as pd
import numpy as np
import random

def get_course_list(date):
	#date = read_data.Date('data/date.csv')

	return date.course_ids

def userid_courseid(enrollment):
	data = pd.read_csv('data/train/enrollment_train.csv',header=0)
	#enrollment = read_data.Enrollment('data/train/enrollment_train.csv')
	num_course_take = []

	for row in data['username']:
		num_course_take.append(len(enrollment.user_info[row]))

	#print(len(num_course_take))

	data['num_course'] = num_course_take
	#print(data)
	return data


def userid_courseid_test(enrollment):
	data = pd.read_csv('data/test/enrollment_test.csv',header=0)
	#enrollment = read_data.Enrollment('data/train/enrollment_train.csv')
	num_course_take = []

	for row in data['username']:
		num_course_take.append(len(enrollment.user_info[row]))

	#print(len(num_course_take))

	data['num_course'] = num_course_take
	#print(data)
	return data


def add_duration(data,date):
	#date = read_data.Date('data/date.csv')

	duration = []

	i=0

	for row in data['course_id']:
		duration.append(date.course_duration[row][0])
		if date.course_duration[row][0]!=30:
			i = i+1

	data['duration'] = duration

	#print "test test here : ", i
	return data

def num_object(data,object):

	#object = read_data.Object('data/object.csv')

	num_object = []

	for row in data['course_id']:
		num_object.append(len(object.course_module[row]))

	data['num_object'] = num_object

	return data


def drop_ratio(data, truth,date):
	ratio = {}
	drop = []

	#truth = read_data.Truth('data/train/truth_train.csv')

	course = get_course_list(date)

	for i in course:
		list1 = list(np.where(data['course_id']==i))
		list2 = list1[0]
		count = 0
		for j in list2:
			enrollment = data.at[j,'enrollment_id']
			

			if truth.truth_info[str(enrollment)][0]=='1':
				count = count + 1
		
		r = count * 100 / len(list2)

		ratio[i] = r


	for row in data['course_id']:
		drop.append(ratio[row])

	data['drop_ratio'] = drop


	#for row1,row2 in zip(data['enrollment_id'],data['course_id']):
		
	return data


def source_event(data,log,date):

	#log = pd.read_csv('data/train/log_train.csv')

	x = 1

	y1 = {}
	y2 = {}
	y3 = {}
	y4 = {}
	y5 = {}
	y6 = {}
	y7 = {}
	y8 = {}
	y9 = {}

	z1 = []
	z2 = []
	z3 = []
	z4 = []
	z5 = []
	z6 = []
	z7 = []
	z8 = []
	z9 = []

	access_browser = 0
	access_server = 0
	discussion_server = 0
	navigate_server = 0
	page_close_browser = 0
	problem_browser = 0
	problem_server = 0
	video_browser = 0
	wiki_server = 0

	t = ''
	t1 = {}
	t2 = []
	t3 = {}
	t4 = []

	for enrollment, time, source, event in zip(log['enrollment_id'],log['time'], log['source'],log['event']):
		if enrollment==1:
			t1[enrollment] = time[:11]

		#if new enrollment appear in log csv, update previous, set all count to 0!
		if x != enrollment:
			#print(x)
			t1[enrollment] = time[:11]

			t3[x] = t[:11]

			y1[x] = access_browser
			y2[x] = access_server
			y3[x] = discussion_server
			y4[x] = navigate_server
			y5[x] = page_close_browser
			y6[x] = problem_browser
			y7[x] = problem_server
			y8[x] = video_browser
			y9[x] = wiki_server

			x = enrollment

			access_browser = 0
			access_server = 0
			discussion_server = 0
			navigate_server = 0
			page_close_browser = 0
			problem_browser = 0
			problem_server = 0
			video_browser = 0
			wiki_server = 0

		t = time

		if source=="browser" and event=="access":
			access_browser += 1
		elif source=="server" and event=="access":
			access_server += 1
		elif source=="server" and event=="discussion":
			discussion_server += 1
		elif source=="server" and event=="navigate":
			navigate_server += 1
		elif source=="browser" and event=="page_close":
			page_close_browser += 1
		elif source=="browser" and event=="problem":
			problem_browser += 1
		elif source=="server" and event=="problem":
			problem_server += 1
		elif source=="browser" and event=="video":
			video_browser += 1
		elif source=="server" and event=="wiki":
			wiki_server += 1
		else:
			print("error here!")

	#translate the time into duration and put it into a list
	#date = read_data.Date('data/date.csv')
	for row1, row2 in zip(data['enrollment_id'],data['course_id']):
		try:
			time_s = date.start[row2]
			duration_s = read_data.get_duration(time_s,t1[row1])
			t2.append(duration_s)
		except:
			t2.append(0)



		try:
			time_e = date.end[row2]
			duration_e = read_data.get_duration(t3[row1],time_e)
			t4.append(duration_e)
		except:
			t4.append(0)
		


	for row in data['enrollment_id']:
		#print(type(row))

		try:
			z1.append(y1[row])
		except:
			z1.append(0)

		try:
			z2.append(y2[row])
		except:
			z2.append(0)

		try:
			z3.append(y3[row])
		except:
			z3.append(0)

		try:
			z4.append(y4[row])
		except:
			z4.append(0)

		try:
			z5.append(y5[row])
		except:
			z5.append(0)

		try:
			z6.append(y6[row])
		except:
			z6.append(0)

		try:
			z7.append(y7[row])
		except:
			z7.append(0)

		try:
			z8.append(y8[row])
		except:
			z8.append(0)

		try:
			z9.append(y9[row])
		except:
			z9.append(0)


	data['access_browser'] = z1
	data['access_server'] = z2
	data['discussion_server'] = z3
	data['navigate_server'] = z4
	data['page_close_browser'] = z5
	data['problem_browser'] = z6
	data['problem_server'] = z7
	data['video_browser'] = z8
	data['wiki_server'] = z9

	data['duration_start'] = t2
	data['duration_end'] = t4

	#print(data)

	return data


#find out average operations of users on different courses
def ops(data,enrollment,log,truth):

	result = []
	result2 = []

	ops = {}
	ops2 = {}
	ops3 = {}

	#enrollment = read_data.Enrollment('data/train/enrollment_train.csv')

	#log = pd.read_csv('data/train/log_train.csv')

	#truth = read_data.Truth('data/train/truth_train.csv')

	x = 1
	count = 0
	y = 0
	yn = 0
	yd = 0

	ops_n = {}
	ops_d = {}
	ops_n2 = {}
	ops_d2 = {}
	result3 = []
	result4 = []


	for row in log['enrollment_id']:
		if x==1:
			x = row

		if row != x:

			ops[x] = count
			
			if truth.truth_info[str(x)][0]=='1':
				ops_n[x] = 0
				ops_d[x] = count
			else:
				ops_n[x] = count
				ops_d[x] = 0				
			
			x = row
			count = 0
		
		count = count + 1


	for row in data['course_id']:
		y = 0
		yn = 0
		yd = 0
		for i in enrollment.course_enrollment[row]:
			try:
				y = y + ops[int(i)]
				yn = yn + ops_n[int(i)]
				yd = yd + ops_d[int(i)]
			except:
				y = y
				yn = yn
				yd = yd


		y = y / len(enrollment.course_info[row])
		z = 0
		for j in enrollment.course_enrollment[row]:
			if truth.truth_info[j][0]=='0':
				z = z+1
		try:
			yn = yn / z
		except:
			yn = 0

		try:
			yd = yd / (len(enrollment.course_enrollment[row]) - z)
		except:
			yd = 0

		ops3[row] = y
		ops_n2[row] = yn
		ops_d2[row] = yd

		try:
			result2.append(ops3[row])
		except:
			result2.append(0)

		try:
			result3.append(ops_n2[row])
		except:
			result3.append(0)

		try:
			result4.append(ops_d2[row])
		except:
			result4.append(0)


	for row in data['username']:
		y = 0
		for i in enrollment.user_enrollment_id[row]:
			try:
				y = y + ops[int(i)]
			except:
				y = y
		y = y / len(enrollment.user_info[row])
		ops2[row] = y
		try:
			result.append(ops2[row])
		except:
			result.append(0)



	data['average_operations_user'] = result
	data['average_operations_course'] = result2
	data['average_operation_course_nondrop'] = result3
	data['average_operation_course_drop'] = result4

	return data


truth = read_data.Truth('data/train/truth_train.csv')
truth2 = read_data.Truth('data/test/truth_test.csv')


'''
Use line 425-460 if you want to read the csv files again!!!!
'''
# enrollment = read_data.Enrollment('data/train/enrollment_train.csv')
# log = pd.read_csv('data/train/log_train.csv')

# date = read_data.Date('data/date.csv')
# object = read_data.Object('data/object.csv')

# enrollment2 = read_data.Enrollment('data/test/enrollment_test.csv')
# log2 = pd.read_csv('data/test/log_test.csv')

# data = userid_courseid(enrollment)
# data = num_object(data,object)
# data = drop_ratio(data,truth,date)
# data = source_event(data,log,date)
# data = ops(data,enrollment,log,truth)


# data2 = userid_courseid_test(enrollment2)
# data2 = num_object(data2,object)
# data2 = drop_ratio(data2,truth2,date)
# data2 = source_event(data2,log2,date)
# data2 = ops(data2,enrollment2,log2,truth2)


# s = pd.Series(data['course_id'])
# f = pd.factorize(s)
# data['course_id'] = f[0]
# s = pd.Series(data['username'])
# f = pd.factorize(s)
# data['username'] = f[0]

# s2 = pd.Series(data2['course_id'])
# f2 = pd.factorize(s2)
# data2['course_id'] = f2[0]
# s2 = pd.Series(data2['username'])
# f2 = pd.factorize(s2)
# data2['username'] = f2[0]


# data.to_csv('preprocess_data.csv',index=False)
# data2.to_csv('preprocess_data_test.csv',index=False)



#change the train dataset into list
#comment this 2 lines if you want to read csv files again
data2 = pd.read_csv('preprocess_data_test.csv')
data = pd.read_csv('preprocess_data.csv')



data = data.values.tolist()
label = truth.truth_label


data2 = data2.values.tolist()
label2 = truth2.truth_label


#####bonus part !!!!    resampling the data to deal with imbalance!!!!

# drop = 0
# nondrop=0
# drop_l = []
# for i in range(len(truth.truth_label)):
# 	if truth.truth_label[i] =='1':
# 		drop+=1
# 		drop_l.append(i)
# 	else:
# 		nondrop+=1

# x = drop-nondrop

# y = random.sample(drop_l,x)

# data = data.drop(y)


# label = truth.truth_label

# for i in range(len(label),0,-1):
# 	if i in y:
# 		del(label[i])

# data = data.values.tolist()






